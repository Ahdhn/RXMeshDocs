# **Solvers**

RXMesh provides a set of linear system solvers that work directly with `SparseMatrix` and `DenseMatrix`. These solvers are built on top of NVIDIA's cuSolver, cuSparse, and optionally cuDSS libraries. They are designed for the types of linear systems that arise in geometry processing—Laplacian systems, Poisson equations, deformation energy minimization, and more.

All solvers follow a common two-phase pattern:

1. **`pre_solve`** — Analyze the matrix structure, compute reordering/permutation, and factorize (for direct solvers) or allocate workspace (for iterative solvers).
2. **`solve`** — Solve the system `AX = B` where `A` is a `SparseMatrix`, `B` is the right-hand side, and `X` is the solution. Both `B` and `X` are `DenseMatrix` objects.

---

## **Choosing a Solver**

| Solver | Type | Best For | Backend |
|--------|------|----------|---------|
| `CholeskySolver` | Direct | Symmetric positive-definite (SPD) systems | cuSolverSp |
| `cuDSSCholeskySolver` | Direct | Large SPD systems | cuDSS |
| `QRSolver` | Direct | General or rectangular systems | cuSolverSp |
| `LUSolver` | Direct | General square systems (host) | cuSolverSp (host API) |
| `CGSolver` | Iterative | SPD systems | cuSparse (SpMV) |
| `PCGSolver` | Iterative | SPD systems (with Jacobi preconditioner) | cuSparse (SpMV) |
| `CGMatFreeSolver` | Iterative | SPD systems (matrix-free) | User-defined matvec |
| `GMGSolver` | Iterative | SPD systems on meshes (multigrid) | Geometric multigrid |

---

## **Direct Solvers**

Direct solvers compute an exact factorization of the matrix. They are well-suited for small-to-medium systems or when high accuracy is needed. All direct solvers support **matrix reordering** to reduce fill-in during factorization.

### Reordering Options

Direct solvers accept a `PermuteMethod` that controls how the matrix is reordered before factorization:

| Method | Description |
|--------|-------------|
| `NONE` | No reordering |
| `SYMRCM` | Symmetric reverse Cuthill-McKee |
| `SYMAMD` | Symmetric approximate minimum degree |
| `NSTDIS` | Nested dissection (METIS, default) |
| `GPUMGND` | GPU-based modified generalized nested dissection |
| `GPUND` | GPU-based nested dissection |

### Cholesky Solver

For symmetric positive-definite systems. This is the recommended direct solver for most geometry processing tasks (e.g., Laplacian smoothing, Poisson solve).

```cpp
SparseMatrix<float> A(rx, Op::VV);
// ... fill A with values ...

DenseMatrix<float> B(rx, rx.get_num_vertices(), 3);
DenseMatrix<float> X(rx, rx.get_num_vertices(), 3);
// ... fill B ...

CholeskySolver<SparseMatrix<float>> solver(&A, PermuteMethod::NSTDIS);
solver.pre_solve(rx);
solver.solve(B, X);
```

The first call to `pre_solve` performs symbolic analysis, permutation, and numerical factorization. If the matrix **values** change but the **sparsity pattern** stays the same, subsequent calls to `pre_solve` will only re-factorize, skipping the symbolic analysis.

### cuDSS Cholesky Solver

An alternative Cholesky solver using NVIDIA's cuDSS library. Requires building with `USE_CUDSS` enabled. Unlike the standard Cholesky solver, it uses a different entry point:

```cpp
cuDSSCholeskySolver<SparseMatrix<float>> solver(&A, PermuteMethod::NSTDIS);
solver.pre_solve(rx, B, X);
solver.solve(B, X);
```

### QR Solver

For general (possibly non-square or non-symmetric) systems:

```cpp
QRSolver<SparseMatrix<float>> solver(&A, PermuteMethod::NSTDIS);
solver.pre_solve(rx);
solver.solve(B, X);
```

### LU Solver

For general square systems. Unlike other direct solvers, this uses a **host-side** API, so the matrix and vectors must be available on the host:

```cpp
LUSolver<SparseMatrix<float>> solver(&A);
A.move(DEVICE, HOST);
B.move(DEVICE, HOST);

solver.pre_solve(rx);
solver.solve(B, X);

X.move(HOST, DEVICE);  // move solution back to GPU if needed
```

---

## **Iterative Solvers**

Iterative solvers approximate the solution through repeated matrix-vector multiplications. They are better suited for large systems where direct factorization would be too expensive.

All iterative solvers accept convergence parameters:

- `max_iter` — Maximum number of iterations.
- `abs_tol` — Absolute residual tolerance.
- `rel_tol` — Relative residual tolerance (compared to initial residual).

After solving, you can query convergence information:

```cpp
solver.iter_taken();      // number of iterations used
solver.final_residual();  // final residual norm
solver.start_residual();  // initial residual norm
```

### CG Solver (Conjugate Gradient)

Unpreconditioned CG for SPD systems:

```cpp
SparseMatrix<float> A(rx, Op::VV);
// ... fill A ...

int unknown_dim = 3;  // number of columns in B and X

DenseMatrix<float> B(rx, rx.get_num_vertices(), unknown_dim);
DenseMatrix<float> X(rx, rx.get_num_vertices(), unknown_dim);
// ... fill B ...

CGSolver<float> cg(A, unknown_dim, /*max_iter=*/1000, /*abs_tol=*/1e-6, /*rel_tol=*/1e-6);
cg.pre_solve(B, X);
cg.solve(B, X);
```

### PCG Solver (Preconditioned CG)

CG with a diagonal (Jacobi) preconditioner. Same interface as `CGSolver`:

```cpp
PCGSolver<float> pcg(A, unknown_dim, /*max_iter=*/1000, /*abs_tol=*/1e-6, /*rel_tol=*/1e-6);
pcg.pre_solve(B, X);
pcg.solve(B, X);
```

### Matrix-Free CG

When the matrix is not explicitly available but you can compute the matrix-vector product, use `CGMatFreeSolver`. You provide a callback that implements the matvec operation:

```cpp
CGMatFreeSolver<float> cg(unknown_dim, /*max_iter=*/1000, /*abs_tol=*/1e-6, /*rel_tol=*/1e-6);

cg.set_mat_vec([&](const DenseMatrix<float>& input, 
                    DenseMatrix<float>& output, 
                    cudaStream_t stream) {
    // implement A * input → output
});

cg.pre_solve(B, X);
cg.solve(B, X);
```

There are also **attribute-based** matrix-free variants (`CGMatFreeAttrSolver`, `PCGMatFreeAttrSolver`) that operate directly on `Attribute` objects instead of `DenseMatrix`, using `ReduceHandle` internally for dot products and norms.

### GMG Solver (Geometric Multigrid)

A multigrid solver that builds a mesh hierarchy for efficient preconditioning. This solver is specific to mesh-based problems and is particularly effective for large Laplacian-type systems:

```cpp
GMGSolver<float> gmg(rx, A, 
    /*max_iter=*/100, 
    /*num_levels=*/4,
    /*num_pre_relax=*/3, 
    /*num_post_relax=*/3, 
    /*coarse_solver=*/CoarseSolver::Cholesky,
    /*sampling=*/Sampling::FPS,
    /*abs_tol=*/1e-6, 
    /*rel_tol=*/1e-6);

gmg.pre_solve(B, X);
gmg.solve(B, X);
```

The GMG solver supports different sampling strategies for building the mesh hierarchy:

| Sampling | Description |
|----------|-------------|
| `Sampling::FPS` | Farthest point sampling (default) |
| `Sampling::Rand` | Random sampling |
| `Sampling::KMeans` | K-means clustering |

---

## **Solver Workflow Summary**

A typical solver workflow:

```cpp
// 1. Build mesh
RXMeshStatic rx("mesh.obj");

// 2. Create sparse matrix from mesh connectivity 
SparseMatrix<float> A(rx, Op::VV);

// 3. Fill matrix values (e.g., cotangent Laplacian)
// ...

// 4. Set up right-hand side and solution
DenseMatrix<float> B(rx, rx.get_num_vertices(), 3);
DenseMatrix<float> X(rx, rx.get_num_vertices(), 3);

// 5. Choose and run a solver
CholeskySolver<SparseMatrix<float>> solver(&A);
solver.pre_solve(rx);
solver.solve(B, X);
```

For systems where the sparsity pattern does not change across iterations (e.g., time-stepping in simulation), you can call `pre_solve` once and then call `solve` repeatedly with updated values in `A` and `B`, re-calling `pre_solve` only when matrix values change and re-factorization is needed.
