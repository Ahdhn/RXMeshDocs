# **Non-linear Solvers**

RXMesh ships four non-linear solvers that sit on top of a [`DiffScalarProblem`](scalar_problem.md) or [`DiffVectorProblem`](vector_problem.md). 

---

## **Choosing a Solver**

| Solver | Problem type | Order | Needs Hessian? | Typical use |
|--------|--------------|-------|----------------|-------------|
| [`NewtonSolver`](#newton) | `DiffScalarProblem` | 2nd | Yes (explicit or matrix-free) | Scalar-valued problems |
| [`GaussNewtonSolver`](#gauss-newton) | `DiffVectorProblem` | ~2nd (JᵀJ) | No (uses Jacobian) | Non-linear least squares |
| [`LBFGSSolver`](#lbfgs) | `DiffScalarProblem` | Quasi-Newton | No | Large problems where assembling a Hessian is too expensive |
| [`GradientDescent`](#gradient-descent) | `DiffScalarProblem` | 1st | No | Simple scalar-valued problems |

All scalar-problem solvers expect the problem's derivatives (`grad`/`hess`/`jac`) to be populated before `compute_direction()`/`take_step()`. The common loop is:

```cpp
problem.eval_terms();       // or eval_terms_grad_only
solver.compute_direction(); // does not apply for GradientDescent
solver.apply_bc(bc_mask);   // optional
solver.line_search(...);    // or take_step() for GradientDescent
```

Linear systems inside `NewtonSolver` and `GaussNewtonSolver` are delegated to the [linear solvers](../matrix/solvers.md). Any of `CholeskySolver`, `cuDSSCholeskySolver`, `LUSolver`, `QRSolver`, `CGSolver`, `PCGSolver`, or `CGMatFreeSolver` can be plugged in.

---

## **Newton** { #newton }

`NewtonSolver` solves `H d = -g` for the search direction `d`, where `H` and `g` are the problem's Hessian and gradient. After that, we usually take a step using line search. 

```cpp
DiffScalarProblem<float, 3, VertexHandle> problem(rx, /*assemble_hessian=*/true);
// ... add terms ...

CholeskySolver<HessianSparseMatrix<float, 3>> linear(problem.hess);
NewtonSolver newton(problem, &linear);

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms();
    newton.compute_direction();
    newton.line_search(/*s_max=*/1.0f, /*shrink=*/0.5f,
                       /*max_iters=*/32, /*armijo_const=*/1e-4f);
}
```

??? note "`NewtonSolver(problem, linear_solver*)`"
    Binds to a `DiffScalarProblem` and a linear solver whose matrix argument is the problem's Hessian. When the linear solver is a `CGMatFreeSolver`, the constructor wires its `mat_vec` callback to `problem.eval_matvec`, so you do not need to assemble the Hessian explicitly.

??? note "`compute_direction()`"
    Calls `linear_solver->pre_solve()` and `solve()` with `-grad` as the right-hand side. The result is stored in an internal `dir` member.

??? note "`apply_bc(bc_attribute)`"
    Masks rows/columns corresponding to fixed degrees of freedom before solving. `bc_attribute` is an `Attribute<int, OptVarHandleT>` where a non-zero entry marks a pinned element. See [Boundary Conditions](advanced.md#boundary-conditions).

??? note "`line_search(s_max, shrink, max_iters, armijo_const, callback = {}, stream = 0)`"
    Backtracking line search with [Armijo](#line-search-and-armijo) sufficient-decrease. `s_max` is the initial step size, `shrink` the backtracking factor, `max_iters` caps the number of trials, and `armijo_const` is the Armijo constant (commonly `1e-4`). The optional `callback(temp_opt_var)` is invoked on the trial iterate before each passive evaluation, useful for recomputing auxiliary state (contact pairs, rotations, ...) — see [line-search callbacks](advanced.md#line-search-callbacks).

### **Matrix-free Newton** { #matrix-free-newton }

When the Hessian is too large to materialize, construct the problem with `assemble_hessian = false` and pair Newton with a `CGMatFreeSolver`:

```cpp
DiffScalarProblem<float, 3, VertexHandle> problem(rx, /*assemble_hessian=*/false);
// ... add terms, using a Scalar type with WithHessian = true so eval_matvec works ...

CGMatFreeSolver<float> cg(hess.rows(), /*unknown_dim=*/3, /*max_iter=*/500, 
                          /*abs_tol*/1e-6f, /*rel_tol*/ 1e-6f);
NewtonSolver newton(problem, &cg);

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms_grad_only(); // or just problem.eval_terms();
    newton.compute_direction();    // Hessian-vector products done on the fly.
    newton.line_search(...);
}
```

---

## **Gauss-Newton** { #gauss-newton }

`GaussNewtonSolver` should be used with [`DiffVectorProblem`](vector_problem.md). It approximates the Hessian with `JᵀJ`, builds that matrix via cuSPARSE SpGEMM, and solves `JᵀJ d = grad` with the linear solver you pick.

```cpp
DiffVectorProblem<float, 12, VertexHandle> problem(rx);
// ... add terms ...
problem.prep_eval();

GaussNewtonSolver gn(problem);
gn.prep_solver(/*cg_max_iter=*/200, /*cg_abs_tol=*/1e-6f, /*cg_rel_tol=*/1e-6f);

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms();
    problem.eval_terms_sum_of_squares(-1.0f);   // grad = -J^T r
    gn.compute_direction();
    // ... step manually or with a line search ...
}
```

??? note "`GaussNewtonSolver(problem)`"
    Binds to a `DiffVectorProblem` and owns the `dir` output buffer that receives the search direction.

??? note "`prep_solver(cg_max_iter, cg_abs_tol, cg_rel_tol)`"
    Allocates `dir` and builds the `JᵀJ` sparse matrix via cuSPARSE SpGEMM with buffer reuse. Call once after `problem.prep_eval()`.

??? note "`compute_direction()`"
    Fills `JᵀJ`, solves `JᵀJ d = grad`, and stores the result in `dir`.

??? note "`release()`"
    Frees the internal `JᵀJ` buffers. Useful when reusing the problem across distinct setups.

Unlike `NewtonSolver`, `GaussNewtonSolver` does not provide `line_search` out of the box — apps typically apply a direct step and, when needed, implement a manual line search using `problem.eval_terms_passive()` and `problem.get_current_loss()`.

---

## **LBFGS** { #lbfgs }

`LBFGSSolver` is a quasi-Newton method that approximates the inverse Hessian from a rolling history of gradient and step differences. It is a good default when assembling a Hessian is not practical and matrix-free Newton convergence is unsatisfactory.

```cpp
DiffScalarProblem<float, 3, VertexHandle, /*WithHess=*/false> problem(rx, false);
// ... add terms ...

LBFGSSolver lbfgs(problem, /*history_size=*/10);

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms_grad_only();
    lbfgs.compute_direction();
    lbfgs.line_search(/*s_max=*/1.0f, /*shrink=*/0.5f,
                      /*max_iters=*/64, /*armijo_const=*/1e-4f);
}
```

??? note "`LBFGSSolver(problem, history_size)`"
    Allocates storage for `history_size` pairs of `(s_k, y_k)` vectors used in the two-loop recursion.

??? note "`compute_direction()`"
    Runs the L-BFGS two-loop recursion over the current history and stores the direction internally.

??? note "`line_search(s_max, shrink, max_iters, armijo_const, stream = 0)`"
    Backtracking line search with [Armijo](#line-search-and-armijo) condition.

---

## **Gradient Descent** { #gradient-descent }

This is the simplest solver option with fixed-learning-rate step in the direction of `-grad`. Best when paired with a `DiffScalarProblem` configured without a Hessian.

```cpp
DiffScalarProblem<float, 3, VertexHandle, /*WithHess=*/false> problem(rx, false);
// ... add terms ...

GradientDescent gd(problem, /*lr=*/1e-3);

for (int it = 0; it < max_iters; ++it) {
    problem.eval_terms();
    gd.take_step();
}
```

??? note "`GradientDescent(problem, lr)`"
    Binds to a `DiffScalarProblem`. `lr` is the scalar learning rate applied to every step.

??? note "`take_step()`"
    Implements `opt_var -= lr * grad` using `for_each` over `OptVarHandleT`. No line search.

---

## **Line Search and Armijo** { #line-search-and-armijo }

The Armijo condition is the standard sufficient-decrease rule used by `NewtonSolver::line_search` and `LBFGSSolver::line_search`. A step `x + s * d` is accepted when

```
f(x + s * d)  ≤  f(x) + armijo_const * s * (∇f · d)
```

Both solvers start at `s = s_max`, shrink by `shrink` on rejection, and give up after `max_iters` trials. Under the hood they call `problem.eval_terms_passive()` on a temporary attribute to evaluate `f(x + s * d)` and `problem.get_current_loss()` to read it.

```cpp
bool accepted = armijo_condition(f_curr, f_new, s, dir, grad, armijo_const);
```

??? note "`armijo_condition<T, Order>(f_curr, f_new, s, dir, grad, armijo_const)`"
    Returns `true` when the Armijo sufficient-decrease inequality holds. `dir` and `grad` are `DenseMatrix`-compatible inputs and must have matching layout (`Order = Eigen::RowMajor` for the scalar problem's `grad`). Typically you do not call this directly, but it is exposed for custom line-search implementations.