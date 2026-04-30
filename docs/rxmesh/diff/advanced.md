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

The block-sparse Hessian owned by [`DiffScalarProblem`](scalar_problem.md) is a `HessianSparseMatrix<T, K>`. Its sparsity pattern is driven from `Op::VV` adjacency where each non-zero is a `K × K` block of second derivatives between two elements. It inherits from [`SparseMatrix`](../sparse_matrices.md), so everything documented there (CSR accessors, `multiply`, `transpose`, `to_eigen`, ...) applies.

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
