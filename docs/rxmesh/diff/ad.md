# **Automatic Differentiation**

RXMesh includes a forward-mode automatic differentiation system for computing gradients, sparse Hessians, and sparse Jacobian of functions/energies defined on a triangle mesh. It is designed to fit the same programming model used throughout RXMesh, i.e., you express your energy as a device lambda over a mesh stencil (`Op::FV`, `Op::EV`, ...) and RXMesh assembles the global gradient, Hessian, or Jacobian.

Most of the machinery already documented in the rest of the library, i.e., [`DenseMatrix`](../matrix/dense_matrices.md), [`SparseMatrix`](../matrix/sparse_matrices.md), [linear solvers](../matrix/solvers.md), and [`for_each`](../static/for_each.md), is reused under the hood. The AD layer adds one new arithmetic primitive (the [`Scalar`](scalar.md) dual number) and two problem types that orchestrate assembly and evaluation.

Here is how a gradient-descent minimization looks like:

```cpp
using namespace rxmesh;

RXMeshStatic rx("mesh.obj");

// Unknowns: 3 floats per vertex (3D positions).
DiffScalarProblem<float, 3, VertexHandle> problem(rx);

// Initialize opt_var from the input coordinates.
problem.opt_var->copy_from(*rx.get_input_vertex_coordinates());

// Add a per-edge term: e(edge) = ||v0 - v1||^2.
problem.add_term<Op::EV>(
    [=] __device__(const auto& eh, const auto& iter, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(eh);

        //Load the vertex position as "active" variable, i.e., a variable with 
        //derivative tracking
        Eigen::Vector3<ActiveT> v0 = opt_var.template active<3>(eh, iter, 0);
        Eigen::Vector3<ActiveT> v1 = opt_var.template active<3>(eh, iter, 1);

        return (v0 - v1).squaredNorm();
    });

GradientDescent gd(problem, /*lr=*/1e-3);

for (int it = 0; it < 100; ++it) {
    problem.eval_terms();
    gd.take_step();
}
```

The optimized positions are now in `*problem.opt_var`. To visualize them, move the attribute to the host and hand it to Polyscope, see [Visualization](../static/visualization.md).

---

## **Mental Model**

You write your objective as a sum of **terms**. Each term is a device lambda invoked per mesh element (per edge, per face, ...) that returns a value. During the active pass, the lambda operates on `Scalar` dual numbers instead of plain floats, so every arithmetic operation carries its derivative with respect to the element's local variables. RXMesh scatters those local derivatives into a global gradient (a [`DenseMatrix`](../matrix/dense_matrices.md)) and, when enabled, a global Hessian (a `HessianSparseMatrix`) or Jacobian (a `JacobianSparseMatrix`).

The unknowns being optimized live in an RXMesh [attribute](../static/working_with_attributes.md) owned by the problem, called `opt_var`. You initialize it from your rest state / embedding / starting point, iterate with a solver, and read the result back from the same attribute.

---

## **Scalar vs. Vector-valued Problems**

RXMesh offers two problem types that correspond to the two most common shapes of geometry-processing objectives:

| Problem type | Objective shape | Derivatives assembled | Solvers |
|--------------|-----------------|-----------------------|-----------------|
| [`DiffScalarProblem`](scalar_problem.md) | `E(x) = Σ e_i(x)` | Gradient and (optional) sparse Hessian | [Newton](nonlinear_solvers.md#newton), [LBFGS](nonlinear_solvers.md#lbfgs), [Gradient Descent](nonlinear_solvers.md#gradient-descent) |
| [`DiffVectorProblem`](vector_problem.md) | `½ ‖r(x)‖²` (stacked residuals) | Jacobian and `Jᵀr` as gradient | [Gauss-Newton](nonlinear_solvers.md#gauss-newton) |

Use the scalar problem when your energy is a sum of scalars (ARAP, Dirichlet, Neo-Hookean, parameterization distortion, ...). Use the vector problem when your objective is a non-linear least squares problem and you want to, e.g., use the `JᵀJ` Gauss-Newton approximation.

---

The following sections cover:

- **[Dual Numbers](scalar.md)**: the arithmetic type used for compute derivatives.
- **[Scalar Problem](scalar_problem.md)**: `DiffScalarProblem`, its `objective`/`grad`/`hess`, and the evaluation API.
- **[Vector Problem](vector_problem.md)**: `DiffVectorProblem` for non-linear least squares.
- **[Terms](terms.md)**: more in-depth discussion about the lambda that defines one additive piece of an objective.
- **[Non-linear Solvers](nonlinear_solvers.md)**: Newton, Gauss-Newton, LBFGS, and gradient descent, including line search.
- **[Advanced Topics](advanced.md)**: Hessian PSD projection, interaction/contact terms, boundary conditions, and line-search callbacks.
