# **Scalar Problem**

`DiffScalarProblem` is the main entry point when your objective is a **sum of scalar energies** over mesh stencils:

```
E(x) = Σ_i  e_i(x)
```

where each `e_i` is one [term](terms.md). The problem owns the optimization variable (`opt_var`), the global gradient (`grad`), and optionally the global Hessian (`hess`). It also orchestrates the active and passive evaluation passes that non-linear solvers drive.

---

## **Construction**

```cpp
template <typename T,          
          int      VariableDim,
          typename OptVarHandleT,
          bool     WithHess = true>
class DiffScalarProblem;
```

- **`T`**: underlying scalar type (`float` or `double`).
- **`OptVarHandleT`**: the mesh element the unknowns live on (usually `VertexHandle`).
- **`VariableDim`**: number of components per element (e.g., `3` for 3D positions, `2` for UVs).
- **`WithHess`**: whether the `Scalar` type used by terms carries Hessian information.

??? info "When to turn off the Hessian"
    Pass `WithHessian = false` (or set `assemble_hessian = false` on the problem) when you only need first-order information, e.g., gradient descent or matrix-free CG Newton. This avoids computing and storing the `k × k` Hessian blocks in every active evaluation and can be significantly faster for large stencils.


```cpp
DiffScalarProblem<float, VertexHandle, 3> problem(rx, /*assemble_hessian=*/true);
```

??? note "`DiffScalarProblem(rx, assemble_hessian, expected_vv_pairs = 0, expected_vf_pairs = 0)`"
    - **`rx`**: the [`RXMeshStatic`](../static.md) instance the problem is tied to.
    - **`assemble_hessian`**: when `true`, allocates `hess` with the `Op::VV` sparsity pattern of `OptVarHandleT` and assembles the global Hessian during `eval_terms()`. Set to `false` for first-order methods ([gradient descent](nonlinear_solvers.md#gradient-descent)) or [matrix-free Newton](nonlinear_solvers.md#matrix-free-newton).
    - **`expected_vv_pairs`** / **`expected_vf_pairs`**: upper bounds on the number of candidate pairs for [interaction terms](advanced.md#interaction-terms). Use `0` when you do not need them.

---

## **The `opt_var` Attribute**

```cpp
std::shared_ptr<Attribute<T, OptVarHandleT>> opt_var;
```

`problem.opt_var` is an RXMesh [attribute](../working_with_attributes.md) with `VariableDim` components per element. It holds the **current iterate**, i.e., the values your optimization is updating. You initialize it before the first evaluation, and you read the final answer from it after the solver finishes.

Typical initialization patterns:

```cpp
// From the input coordinates (e.g., deformation solvers).
problem.opt_var->copy_from(*rx.get_input_vertex_coordinates());

// From a 2D embedding (e.g., parameterization).
rx.for_each_vertex(DEVICE,
    [opt_var = *problem.opt_var, init_uv] __device__(const VertexHandle& vh) {
        opt_var(vh, 0) = init_uv(vh, 0);
        opt_var(vh, 1) = init_uv(vh, 1);
    });
```

After the solver has converged, the updated values are in the same attribute. To visualize or export them, move the attribute to the host (see [Working with Attributes](../working_with_attributes.md) and [Visualization](../visualization.md)):

```cpp
problem.opt_var->move(DEVICE, HOST);
```

---

## **Gradient and Hessian**

```cpp
DenseMatrix<T, Eigen::RowMajor> grad;               // (|elems|, VariableDim)
HessianSparseMatrix<T, VariableDim>* hess = nullptr; // optional
```

- **`grad`** is a [`DenseMatrix`](../dense_matrices.md) whose `i`-th row holds the global gradient entries at element `i`. Indexable by both linear index and [`OptVarHandleT`](../handles.md).
- **`hess`** is a block-sparse matrix whose non-zero structure is the `Op::VV` adjacency of `OptVarHandleT`. Each non-zero is a `VariableDim × VariableDim` block of second derivatives. It is only allocated when `assemble_hessian = true`. Access via `(row_handle, col_handle, local_i, local_j)`; see [Advanced Topics](advanced.md#hessian-matrix) for the block access API. The underlying container inherits from [`SparseMatrix`](../sparse_matrices.md), so matrix operations like `multiply` and `transpose` are available.

Both `grad` and `hess` are zeroed at the start of every call to `eval_terms()`.

---

## **Adding Terms**

```cpp
problem.add_term<Op, ProjectHess, blockThreads>(lambda, oriented = false);
```

- **`Op`**: the mesh query that defines the stencil of this term (see [supported query types](../for_each.md#supported-query-types)).
- **`ProjectHess`**: boolean for whether to project the local Hessian to be Positive semidefinite matrix. See [Hessian PSD Projection](./advanced.md#hessian-psd-projection)
- **`blockThreads`**: the CUDA block size 
- **`lambda`**: the device lambda described in [Terms](terms.md#the-term-lambda).
- **`oriented`**: forwarded to the underlying query (only meaningful for some ops, e.g., `Op::VV`, `Op::VE`).

Two examples:

```cpp
// Per-face area energy (Op::FV, 3 vertices * 3 coords = 9 local vars).
problem.add_term<Op::FV, true>(
    [=] __device__(const auto& fh, const auto& fv, auto& opt_var) {
        auto p0 = opt_var.template active<3>(fh, fv, 0);
        auto p1 = opt_var.template active<3>(fh, fv, 1);
        auto p2 = opt_var.template active<3>(fh, fv, 2);
        auto n  = (p1 - p0).cross(p2 - p0);
        return 0.5f * sqrt(n.dot(n));
    });

// Per-edge spring (Op::EV, 2 vertices * 3 coords = 6 local vars).
problem.add_term<Op::EV, true>(
    [rest_len = *rest_len_attr] __device__(
        const auto& eh, const auto& ev, auto& opt_var) {
        auto v0 = opt_var.template active<3>(eh, ev, 0);
        auto v1 = opt_var.template active<3>(eh, ev, 1);
        auto d  = (v0 - v1).norm() - rest_len(eh);
        return d * d;
    });
```

You can add as many terms as you need. All terms will be summed into the same `grad` / `hess` during evaluation.

??? note "`add_term<Op, bool, uint32_t, typename LambdaT>(lambda, oriented = false)`"
    Registers one additive scalar term. `lambda` is a device callable that returns `ScalarT` (active pass) or its passive equivalent (passive pass). The corresponding per-term loss attribute and reducer are created internally so the term can contribute to `get_current_loss()`.

??? note "`add_interaction_term<Op, bool, uint32_t, typename LambdaT>(lambda)`"
    Variant for **candidate-pair** based terms, supported for `Op::VV` and `Op::VF`. See [Advanced Topics](advanced.md#interaction-terms).

??? note "`update_hessian()`"
    Call after changing the set of candidate pairs used by interaction terms. The Hessian sparsity pattern is re-derived and `hess` is re-allocated to match. This is not needed to be called with the terms are added only via the `add_term` API.

---

## **Evaluation**

A solver step is typically:

```cpp
problem.eval_terms();           // fills grad (and hess if enabled)
solver.compute_direction();
solver.line_search(...);        // uses eval_terms_passive internally
```

```cpp
// Active pass: gradient (and Hessian if assembled).
problem.eval_terms(cudaStream_t stream = 0);

// Active pass: gradient only, even if hess is allocated.
problem.eval_terms_grad_only(cudaStream_t stream = 0);

// Passive pass: evaluate energies at `opt_var` (or at `*problem.opt_var` if not null),
// updating per-term loss so get_current_loss() returns the new value.
problem.eval_terms_passive(Attribute<T, OptVarHandleT>* opt_var = nullptr,
                           cudaStream_t stream = 0);

// Current total energy across all terms.
T loss = problem.get_current_loss(cudaStream_t stream = 0);

// Hessian-vector product: out = hess * in. Requires WithHess + assemble_hessian.
problem.eval_matvec(in, out);
```

??? note "`eval_terms(stream = 0)`"
    Active pass. Resets `grad` (and `hess`) to zero, launches each term's active kernel, and accumulates into the global derivatives. Also updates per-term losses so `get_current_loss()` reflects the current iterate.

??? note "`eval_terms_grad_only(stream = 0)`"
    Same as `eval_terms()` but skips the Hessian assembly even when `assemble_hessian = true`.

??? note "`eval_terms_passive(opt_var = nullptr, stream = 0)`"
    Passive pass over `opt_var` (defaults to `*problem.opt_var`). No derivatives, only the per-term losses are updated. Pair this with `get_current_loss()` when writing a manual line search or evaluating trial iterates.

??? note "`eval_matvec(in, out, stream = 0)`"
    Applies the Hessian to a `DenseMatrix` column (`out = hess * in`). Only valid when the underlying `Scalar` type carries Hessian information. 

??? note "`get_current_loss(stream = 0)`"
    Returns the sum of the per-term loss reductions as a host scalar. Valid after any `eval_terms*` call.

---

## **End-to-End Execution**

Putting the scalar problem together with a [Newton solver](nonlinear_solvers.md#newton) and a [Cholesky linear solver](../solvers.md#cholesky-solver) gives the canonical optimization loop used by in many applications:

```cpp
RXMeshStatic rx("mesh.obj");

DiffScalarProblem<float, 3, VertexHandle> problem(rx, /*assemble_hessian=*/true);

// 1. Initialize opt_var (e.g., from rest shape).
problem.opt_var->copy_from(*rx.get_input_vertex_coordinates());

// 2. Register term(s).
problem.add_term<Op::EV, true>(
    [=] __device__(const auto& eh, const auto& iter, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(eh);
        ActiveT e;
        //compute e
        //....
        return e;
    });

// 3. Build a linear solver and wrap it in Newton.
CholeskySolver<HessianSparseMatrix<float, 3>> linear(problem.hess);
NewtonSolver newton(problem, &linear);

// 4. Iterate.
for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms();
    if (problem.grad.norm2() < tol) break;

    newton.compute_direction();
    newton.apply_bc(bc_mask);       // optional; see Advanced Topics.
    newton.line_search(/*s_max=*/1.0f, 
                       /*shrink=*/0.5f, 
                       /*max_iters=*/32,
                       /*armijo_const=*/1e-4f);
}

// 5. Read results.
problem.opt_var->move(DEVICE, HOST);
```

For the vector variant, see [Vector Problem](vector_problem.md). For alternative solvers (`LBFGS`, `GradientDescent`, matrix-free Newton), see [Non-linear Solvers](nonlinear_solvers.md).
