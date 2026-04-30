# **Vector Problem**

`DiffVectorProblem` is the entry point for **non-linear least squares** objectives, i.e., energies that are written as the squared norm of a residual vector:

```
E(x) = ½ ‖r(x)‖²     with     r(x) = [ r_1(x); r_2(x); ... ]
```

Each term contributes a block of residual rows, and RXMesh assembles the global Jacobian `J = ∂r/∂x` in sparse form. A [Gauss-Newton solver](nonlinear_solvers.md#gauss-newton) then can be used which uses the approximation `H ≈ JᵀJ` to compute search directions. Use this problem type when your objective is a sum of squared residuals, e.g., geometric fitting, regularized rotations, or frame-field alignment.

The user-facing shape of a vector problem mirrors [`DiffScalarProblem`](scalar_problem.md), i.e., the same `opt_var` attribute, the same [term lambda pattern](terms.md#the-term-lambda), and the same notion of active/passive evaluation. The differences are (a) terms return an Eigen vector instead of a scalar, and (b) derivative storage is a Jacobian, not a Hessian.

---

## **Construction**

```cpp
template <typename T,          
          int      VariableDim,
          typename OptVarHandleT>
class DiffVectorProblem;
```

- **`T`**: underlying scalar type (`float` or `double`).
- **`VariableDim`**: number of components per element.
- **`OptVarHandleT`**: mesh element the unknowns live on.


```cpp
DiffVectorProblem<float, 12, VertexHandle> problem(rx);
```

??? note "`DiffVectorProblem(rx)`"
    Creates the `opt_var` attribute with `VariableDim` components per element of type `OptVarHandleT`. The Jacobian, residual vector, and gradient are only allocated later by `prep_eval()`, because their shape depends on the terms you register.

As with the scalar problem, `problem.opt_var` is the attribute that holds the current iterate. You initialize it before the first evaluation and read the result from it after convergence.

---

## **Adding Terms**

A vector term returns an `Eigen::Matrix<ActiveT, OutputDim, 1>`. Each invocation of the lambda contributes one block of `OutputDim x VariableDim` residual:

```cpp
problem.add_term<Op::EV, 7>(
    [=] __device__(const auto& eh, const auto& ev, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(eh);

        // 7-component residual
        Eigen::Vector<ActiveT, 7> res;

        auto v0 = opt_var.template active<3>(eh, ev, 0);
        auto v1 = opt_var.template active<3>(eh, ev, 1);
        
        //calculate res different components, r[0], r[1], ..., r[6]
        //.... 
        
        return res;
    });
```

Terms are stacked vertically in the Jacobian in the order they were added. 

??? info "Row ranges per term"
    After `prep_eval()` you can query `problem.jac->get_term_rows_range(i)` to find the row slice produced by term `i`. This is mostly useful for diagnostics and sanity check.

---

## **Jacobian, Residual, Gradient**

```cpp
std::unique_ptr<JacobianSparseMatrix<T>>         jac;        // (|r|, |x|)
std::unique_ptr<DenseMatrix<T>>                  residual;   // (|r|, 1)
DenseMatrix<T, Eigen::RowMajor>                  grad;       // (|elems|, VariableDim)
std::shared_ptr<Attribute<T, OptVarHandleT>>     opt_var;
```

- **`jac`**: a CSR sparse matrix that stacks the per-term block-sparse Jacobians. Inherits from [`SparseMatrix`](../matrix/sparse_matrices.md), so `multiply` and `transpose` work out of the box. Access by `(term_id, row_handle, col_handle, local_i, local_j)`.
- **`residual`**: the stacked residual `r(x)`. Filled by `eval_terms()`.
- **`grad`**: `Jᵀ r`, produced by `eval_terms_sum_of_squares()`. Has the same shape as a `DiffScalarProblem` gradient so the same solver plumbing can consume it.

These members are allocated by `prep_eval()` and are only valid after that call.

---

## **Evaluation**

The vector problem splits its evaluation into two steps so that Gauss-Newton can reuse the Jacobian without recomputing the residual:

```cpp
problem.prep_eval(); // once, after all add_term calls.                    

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms(); // fills residual and jac.
    problem.eval_terms_sum_of_squares(/*scale=*/-1.0);  // grad = -Jᵀ r.

    gn.compute_direction();
    // ... take a step (see Non-linear Solvers) ...
}
```

```cpp
// Allocate jac, residual, grad based on the registered terms.
problem.prep_eval();

// Active pass: fills residual and jac at the current objective.
problem.eval_terms(cudaStream_t stream = 0);

// Compute grad = scale * J^T * residual (typical: scale = -1).
problem.eval_terms_sum_of_squares(T scale = 1, cudaStream_t stream = 0);

// Passive pass on `opt_var` (or *problem.opt_var if null). Updates losses only.
problem.eval_terms_passive(Attribute<T, OptVarHandleT>* opt_var = nullptr,
                           cudaStream_t stream = 0);

// Current energy = ||residual||^2.
T loss = problem.get_current_loss(cudaStream_t stream = 0);
```

??? note "`prep_eval()`"
    Must be called after all `add_term` calls but before the first evaluation. Derives the Jacobian sparsity from the set of registered terms, allocates `jac`, `residual`, and `grad`, and prepares internal buffers (e.g., reducers for the per-term losses).

??? note "`eval_terms(stream = 0)`"
    Active pass. Resets `jac` and `residual` to zero, launches each term kernel, and writes both the residual block and the Jacobian block produced by forward-mode AD.

??? note "`eval_terms_sum_of_squares(scale = 1, stream = 0)`"
    Must be called after `eval_terms()`. Computes `grad = scale * Jᵀ * residual` via `jac->multiply(..., transpose_A=true)`. The typical Gauss-Newton convention is `scale = -1.0` so the linear system to solve becomes `JᵀJ d = grad`.

??? note "`eval_terms_passive(opt_var = nullptr, stream = 0)`"
    Passive pass for line search or trial evaluation. Updates `residual`'s magnitude and the per-term losses without touching the Jacobian.

??? note "`get_current_loss(stream = 0)`"
    Returns `‖residual‖²`. Valid after any `eval_terms*` call.

---

## **End-to-End Execution**

The typical vector-problem loop drives a [Gauss-Newton solver](nonlinear_solvers.md#gauss-newton), as used by [EmbeddedARAP](https://github.com/owensgroup/RXMesh/tree/main/apps/EmbeddedARAP) and [PolyVector](https://github.com/owensgroup/RXMesh/tree/main/apps/PolyVector):

```cpp
RXMeshStatic rx("mesh.obj");

DiffVectorProblem<float, VertexHandle, 12> problem(rx);

// 1. Initialize opt_var.
// ... populate *problem.opt_var with a starting guess ...

// 2. Register residual terms.
problem.add_term<Op::V, 2>(
    [=] __device__(const auto& vh, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(eh);
        Eigen::Vector<ActiveT, 2> e;
        //.... compute e
        //...        
        return e;
        
    });

problem.add_term<Op::EV, 4>(
    [=] __device__(const auto& eh, const auto& ev, auto& opt_var) {        
        Eigen::Vector<ActiveT, 4> e;
        //.... compute e
        //...        
        return e;
    });

// 3. Allocate Jacobian / residual / grad.
problem.prep_eval();

// 4. Build a Gauss-Newton solver on top.
GaussNewtonSolver gn(problem);
gn.prep_solver(/*cg_max_iter=*/200, /*cg_abs_tol=*/1e-6f, /*cg_rel_tol=*/1e-6f);

// 5. Iterate: eval -> direction -> step.
for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms();
    problem.eval_terms_sum_of_squares(-1.0f);

    gn.compute_direction();

    // Simple step; apps may do a manual line search using eval_terms_passive.
    rx.for_each_vertex(DEVICE,
        [opt_var = *problem.opt_var, dir = gn.dir] __device__(const VertexHandle& vh) {
            for (int c = 0; c < 12; ++c) opt_var(vh, c) += dir(vh, c);
        });
}

// 6. Read results.
problem.opt_var->move(DEVICE, HOST);
```

For the scalar variant, see [Scalar Problem](scalar_problem.md). For the solver API and line-search options, see [Non-linear Solvers](nonlinear_solvers.md).
