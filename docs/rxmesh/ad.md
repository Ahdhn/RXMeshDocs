# **Automatic Differentiation**

RXMesh includes a GP forward-mode automatic differentiation system for computing gradients, sparse Hessians, and sparse Jacobian of functions/energies defined on a triangle mesh. It is designed to fit the same programming model used throughout RXMesh, i.e., you express your energy as a device lambda over a mesh stencil (`Op::FV`, `Op::EVDiamond`, ...) and RXMesh assembles the global gradient, Hessian, or Jacobian for you.

Most of the machinery already documented in the rest of the library, i.e., [`DenseMatrix`](dense_matrices.md), [`SparseMatrix`](sparse_matrices.md), [linear solvers](solvers.md), and [`for_each`](for_each.md), is reused under the hood. The AD layer adds one new arithmetic primitive (the [`Scalar`](diff/scalar.md) dual number) and two problem types that orchestrate assembly and evaluation.

Here is what a tiny gradient-descent minimization looks like, the shape of every application is the same, i.e., construct a problem, initialize `objective`, `add_term`, evaluate, step:

```cpp
using namespace rxmesh;

RXMeshStatic rx("mesh.obj");

// Unknowns: 3 floats per vertex (3D positions).
// assemble_hessian = false -> gradient only.
DiffScalarProblem<float, VertexHandle, 3, false> problem(rx, false);

// Initialize objective from the input coordinates.
problem.objective->copy_from(*rx.get_input_vertex_coordinates());

// Add a per-edge term: e(edge) = ||v0 - v1||^2.
problem.add_term<Op::EV>(
    [=] __device__(const auto& eh, const auto& iter, auto& objective) {
        using ActiveT = ACTIVE_TYPE(fh);

        Eigen::Vector3<ActiveT> v0 = iter_val<ActiveT, 3>(eh, iter, objective, 0);
        Eigen::Vector3<ActiveT> v1 = iter_val<ActiveT, 3>(eh, iter, objective, 1);
        auto d  = v0 - v1;
        return d..squaredNorm();
    });

GradientDescent gd(problem, /*lr=*/1e-3);

for (int it = 0; it < 100; ++it) {
    problem.eval_terms();
    gd.take_step();
}
```

The optimized positions are now in `*problem.objective`. To visualize them, move the attribute to the host and hand it to Polyscope, see [Visualization](visualization.md).

---

## **Mental Model**

You write your objective as a sum of **terms**. Each term is a device lambda invoked per mesh element (per edge, per face, ...) that returns a number. During the active pass, the lambda operates on `Scalar` dual numbers instead of plain floats, so every arithmetic operation carries its derivative with respect to the element's local variables. RXMesh scatters those local derivatives into a global gradient (a [`DenseMatrix`](dense_matrices.md)) and, when enabled, a global Hessian (a [`HessianSparseMatrix`](diff/advanced.md)) or Jacobian (a `JacobianSparseMatrix`).

The unknowns being optimized live in an RXMesh [attribute](working_with_attributes.md) owned by the problem, called `objective`. You initialize it from your rest state / embedding / starting point, iterate with a solver, and read the result back from the same attribute.

---

## **Scalar vs. Vector-valued Problems**

RXMesh offers two problem types that correspond to the two most common shapes of geometry-processing objectives:

| Problem type | Objective shape | Derivatives assembled | Typical solvers |
|--------------|-----------------|-----------------------|-----------------|
| [`DiffScalarProblem`](diff/scalar_problem.md) | `E(x) = Σ e_i(x)` | Gradient and (optional) sparse Hessian | [Newton](diff/nonlinear_solvers.md#newton), [LBFGS](diff/nonlinear_solvers.md#lbfgs), [Gradient Descent](diff/nonlinear_solvers.md#gradient-descent) |
| [`DiffVectorProblem`](diff/vector_problem.md) | `½ ‖r(x)‖²` (stacked residuals) | Jacobian and `Jᵀr` as gradient | [Gauss-Newton](diff/nonlinear_solvers.md#gauss-newton) |

Use the scalar problem when your energy is a sum of scalars (ARAP, Dirichlet, Neo-Hookean, parameterization distortion, ...). Use the vector problem when your objective is a non-linear least squares problem and you want to, e.g., use the `JᵀJ` Gauss-Newton approximation.

---

The following sections cover:

- **[Dual Numbers](diff/scalar.md)**: the arithmetic type used for compute derivatives.
- **[Scalar Problem](diff/scalar_problem.md)**: `DiffScalarProblem`, its `objective`/`grad`/`hess`, and the evaluation API.
- **[Vector Problem](diff/vector_problem.md)**: `DiffVectorProblem` for non-linear least squares.
- **[Terms](diff/terms.md)**: more in-depth discussion about the lambda that defines one additive piece of an objective.
- **[Non-linear Solvers](diff/nonlinear_solvers.md)**: Newton, Gauss-Newton, LBFGS, and gradient descent, including line search.
- **[Advanced Topics](diff/advanced.md)**: Hessian PSD projection, interaction/contact terms, boundary conditions, and line-search callbacks.
