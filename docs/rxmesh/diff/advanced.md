# **Advanced Topics**

This page collects the somehow less-common but useful workflows the AD layer supports, e.g., projecting local Hessians to PSD, expressing contact / interaction as candidate-pair terms, pinning degrees of freedom with boundary conditions, refreshing auxiliary state during line search, and a few utility helpers that show up when you write custom terms.

---

## **Hessian PSD Projection** { #hessian-psd-projection }

Newton's method requires the Hessian to be positive definite for the search direction to actually decrease the energy. Non-convex energies (e.g., Neo-Hookean) could produce indefinite local Hessians. One way to fix this is to project each stencil's local Hessian to a nearby PSD matrix before accumulating it into the global Hessian.

The option of projecting local Hessian to PSD is provided as template Boolean to `add_term`

```cpp
problem.add_term<Op::FV, /*ProjectHess*/ true>(
    [=] __device__(const auto& fh, const auto& fv) {
        //....
    });
```

??? info "How the projection works"
    When `ProjectHess = true`, RXMesh makes each local Hessian PSD as follows:

    1. **Cheap early-out via diagonal dominance.** First, the local Hessian is checked for strict diagonal dominance with a positive diagonal, i.e., for every row `i`, `H_ii > sum_{j != i} |H_ij| + eps`. This is a sufficient condition for positive-definiteness, so when it holds the matrix is left untouched and the eigen-decomposition is skipped.
    2. **Eigenvalue clamping.** Otherwise, the symmetric Hessian is eigen-decomposed, every eigenvalue below a small positive floor is clamped to that floor, and the matrix is reassembled as `Q D' Q^T`. This produces the closest PSD matrix. If all eigenvalues were already above the floor, the matrix is left unchanged.



---

## **Interaction / Contact Terms** { #interaction-terms }

Contact and collision energies depend on pairs of mesh elements that are **not** related by the mesh's fixed connectivity. RXMesh models these as **candidate pairs**, i.e., dynamically-managed lists of `(VertexHandle, VertexHandle)` or `(VertexHandle, FaceHandle)` associations that change as the optimization progresses. The user owns the logic that refreshes these pairs (e.g., a BVH proximity query) while RXMesh owns their storage and the resulting Hessian sparsity pattern.

The expected capacity of each list is fixed at problem construction time:

```cpp
DiffScalarProblem<float, 3, VertexHandle> problem(rx,
    /*assemble_hessian=*/true,
    /*expected_vv_pairs=*/max_vv_pairs,
    /*expected_vf_pairs=*/max_vf_pairs);
```

### Workflow { #interaction-three-step }

A Newton iteration that involves contact looks like this:

**Step A: Compute interaction energy (once, at setup).** Register the contact / barrier energies with `problem.add_interaction_term<Op::VV, ProjectHess>(lambda)` and/or `add_interaction_term<Op::VF, ProjectHess>(lambda)`. Like `add_term`, these are registered once and reused on every `eval_terms()` call.

**Step B: Populate candidate pairs (each iteration).** The application fills `problem.vv_pairs` and/or `problem.vf_pairs` with the pairs that should be considered this iteration. The typical pattern is `pairs.reset()` followed by `pairs.insert(h0, h1)` for each detected pair (e.g., from a BVH query). Insertion can happen on the GPU (and the user does not have to worry about race condition during insertion).

**Step C: Refresh sparsity and evaluate.** Call `problem.update_hessian()` so the Hessian sparsity pattern picks up the new pairs, then `problem.eval_terms()` as usual.


```cpp
// --- once, at setup ---
vv_contact_energy(problem, contact_area);  // calls add_interaction_term<Op::VV>
vf_contact_energy(problem, contact_area);  // calls add_interaction_term<Op::VF>

// --- each Newton iteration ---
add_contact(problem, rx,
            problem.vv_pairs, problem.vf_pairs,
            /* BVH buffers, x, contact_area, ... */);  // Step B: fill the pairs
problem.update_hessian();                              // Step C: refresh sparsity
problem.eval_terms();                                  //         then evaluate
```

If the line-search callback also refreshes the candidate pairs (so that trial iterates use up-to-date contacts), call `update_hessian()` again inside the callback before re-evaluating.

### VV Interaction Terms { #interaction-vv }

For vertex-vertex contact, register the term with `Op::VV`. The lambda takes a single seed handle plus the pair iterator:

```cpp
problem.add_interaction_term<Op::VV, /*ProjectHess*/ true>(
    [=] __device__(const auto& id, const auto& iter, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(id);

        const VertexHandle v0 = iter[0];   // first vertex of the pair
        const VertexHandle v1 = iter[1];   // second vertex of the pair

        Eigen::Vector3<ActiveT> xi = opt_var.template active<3>(id, iter, 0);
        Eigen::Vector3<ActiveT> xj = opt_var.template active<3>(id, iter, 1);

        ActiveT d = (xi - xj).norm();
        return /* barrier energy in d */;
    });
```

`id` is only used to deduce `ACTIVE_TYPE(id)`; the actual pair lives in `iter[0]` and `iter[1]`. The local variable vector has size `2 * VariableDim` (one slot per pair member). Real call site: [`vv_contact_energy`](https://github.com/owensgroup/RXMesh/blob/main/apps/NeoHookean/barrier_energy.h).

### VF Interaction Terms { #interaction-vf }

For vertex-face contact, register the term with `Op::VF`. The lambda takes the face handle, the colliding vertex handle, and an iterator that walks the three vertices of the face:

```cpp
problem.add_interaction_term<Op::VF, /*ProjectHess*/ true>(
    [=] __device__(const auto& fh, const auto& vh,
                   const auto& iter, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(fh);

        // slot 0 reads the colliding vertex `vh`;
        // slots 1..3 read the three vertices of the face `fh` (FV stencil).
        Eigen::Vector3<ActiveT> xi = opt_var.template active<3>(fh, vh, iter, 0);
        Eigen::Vector3<ActiveT> p0 = opt_var.template active<3>(fh, vh, iter, 1);
        Eigen::Vector3<ActiveT> p1 = opt_var.template active<3>(fh, vh, iter, 2);
        Eigen::Vector3<ActiveT> p2 = opt_var.template active<3>(fh, vh, iter, 3);

        return /* contact / barrier energy of `xi` against triangle (p0,p1,p2) */;
    });
```

The local variable vector has size `(iter.size() + 1) * VariableDim` = `4 * VariableDim`: one slot for `vh` plus one slot per iterator entry. Real call site: [`vf_contact_energy`](https://github.com/owensgroup/RXMesh/blob/main/apps/NeoHookean/barrier_energy.h).


??? note "`add_interaction_term<Op, ProjectHess>(lambda)`"
    Variant of `add_term` for candidate-pair based terms. Only `Op::VV` and `Op::VF` are supported, and the lambda shape depends on `Op`:

    - `Op::VV`: lambda is `(id, iter, opt_var)`. `iter[0]` and `iter[1]` are the two vertices of the pair.
    - `Op::VF`: lambda is `(fh, vh, iter, opt_var)`. `vh` is the colliding vertex; `iter` traverses the three vertices of `fh` (FV stencil).

    Registered once at setup; reused on every `eval_terms()`.

??? note "`opt_var.active<VariableDim>(id, iter, index_in_iter)` (VV form)"
    Reads `iter[index_in_iter]` and seeds it as an independent variable inside a local vector of size `2 * VariableDim`. 

??? note "`opt_var.active<VariableDim>(fh, vh, iter, index_in_iter)` (VF / mixed stencil form)"
    `index_in_iter == 0` reads `vh`; `index_in_iter == k` for `k >= 1` reads `iter[k - 1]`. The slot at `index_in_iter` is treated as an independent variable inside a local vector of size `(iter.size() + 1) * VariableDim`.

??? note "`problem.vv_pairs`, `problem.vf_pairs`"
    Public `CandidatePairs` members owned by `DiffScalarProblem`. Application code refreshes them each iteration. Common operations: `reset()`, `insert(h0, h1)`, `num_pairs()`.

??? note "`update_hessian()`"
    Rederives the Hessian sparsity pattern from the current `vv_pairs` / `vf_pairs` and reallocates `problem.hess` accordingly. Must be called after changing the pair lists and before the next `eval_terms()`.

See [NeoHookean](https://github.com/owensgroup/RXMesh/tree/main/apps/NeoHookean) for a full contact pipeline.

---

## **Boundary Conditions** { #boundary-conditions }

Pinning part of the mesh is expressed by a **boundary-condition attribute** that marks fixed elements. `NewtonSolver::apply_bc(bc)` zeros the corresponding rows (and columns, for symmetry) of the system before solving, so the pinned values never move.

```cpp
auto bc = *rx.add_vertex_attribute<int>("bc", 1, DEVICE);
// .... populate bc attribute 

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms();
    newton.compute_direction();
    newton.apply_bc(bc);        // zero-out pinned rows/cols
    newton.line_search(...);
}
```

The convention is `bc(h) != 0` means "handle `h` is fixed".

---

## **Line-Search Callbacks** { #line-search-callbacks }

Some energies depend on auxiliary state that must be refreshed whenever the iterate moves (e.g., contact pairs for collision). `NewtonSolver::line_search` accepts an optional callback that fires on each trial iterate before the passive evaluation:

```cpp
newton.line_search(
    /*s_max=*/1.0f, /*shrink=*/0.5f, /*max_iters=*/32, /*armijo_const=*/1e-4f,
    [&] __host__(Attribute<float, VertexHandle>* temp_opt_var) {
        // temp_opt_var holds the trial iterate; recompute state from it.
        refresh_rotations(*temp_opt_var);
    });
```

The callback runs on the host but is free to launch its own kernels. Use this when a naive line search would compare energies computed with stale auxiliary data.

---

## **`util.h` Helpers** { #util }

A handful of small helpers show up in term code and solver:

??? note "`col_mat(v0, v1, ...)`"
    Stacks column vectors into an `Eigen::Matrix`. 

??? note "`is_finite(x)`, `is_nan(x)`, `is_inf(x)`, `is_finite_mat(M)`, `is_finite_scalar(s)`"
    Boolean checks that work on plain scalars, `Scalar` values, and Eigen matrices. Handy guards inside a term to skip problematic stencils without producing NaNs.

??? note "`is_same_matrix(A, B, tol)`, `is_sym(M, tol)`"
    Matrix-equality and symmetry checks against a tolerance.

??? note "`sqr(x)`"
    Convenience `x * x`, overloaded for plain scalars and for `Scalar` / Eigen matrices.
---

## **Hessian Matrix** { #hessian-matrix }

The block-sparse Hessian owned by [`DiffScalarProblem`](scalar_problem.md) is a `HessianSparseMatrix<T, K>`. Its sparsity pattern is driven from `Op::VV` adjacency where each non-zero is a `K × K` block of second derivatives between two elements. It inherits from [`SparseMatrix`](../matrix/sparse_matrices.md), so everything documented there (CSR accessors, `multiply`, `transpose`, `to_eigen`, ...) applies.

The block-aware accessors are the main addition:

??? note "`operator()(row_handle, col_handle, local_i, local_j)`"
    Access entry `(local_i, local_j)` of the `K × K` block at handle pair `(row_handle, col_handle)`.

??? note "`get_indices(row_handle, col_handle, local_i, local_j)`"
    Returns the flat `(row_id, col_id)` a regular `SparseMatrix` accessor would use. This is useful when bridging to external CSR consumers.

??? note "`is_non_zero(row_handle, col_handle)`"
    Checks whether the block at that handle pair is allocated (i.e., the two elements are adjacent in the pattern).

??? note "`HessianSparseMatrix(rx, extra_nnz_entries = 0, op = Op::VV)`"
    Direct constructor if you need a standalone Hessian-style matrix outside a `DiffScalarProblem`. `extra_nnz_entries` reserves space for additional blocks (e.g., for interaction terms).

The Jacobian used by [`DiffVectorProblem`](vector_problem.md) is analogous: `JacobianSparseMatrix<T>` inherits from `SparseMatrix` and adds per-term row-range and block accessors.
