# ** Dense Matrices ** 

RXMesh provides a `DenseMatrix` class designed for working with large numerical data tied to the mesh. `DenseMatrix` are allocated on both host and device and they are compatible with [`cuBLAS`](https://developer.nvidia.com/cublas) (on the device) and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (on the host). 

Dense matrices are often constructed from mesh attributes, e.g., vertex coordinates or physical quantities. Attributes can be directly converted to dense matrices and vice versa (see [Attributes](attributes.md)).

For small matrix operations—e.g., working with 3×3 matrices for SVD—we recommend using the attribute interface's `to_eigen` and `to_glm` methods (also documented in [Attributes](attributes.md)). These are optimized for use within GPU kernels and avoid unnecessary memory transfers.

_Example_: Converting vertex coordinates into a dense matrix, doing computations, then writing them back:

```cpp
RXMeshStatic rx("mesh.obj");

// Get vertex coordinates as attribute
std::shared_ptr<VertexAttribute<float>> x = rx.get_input_vertex_coordinates();

// Convert to (#vertices × 3) dense matrix
std::shared_ptr<DenseMatrix<float>> x_mat = x->to_matrix();

// Do something with x_mat
// ...

// Copy results back to attribute
x->from_matrix(x_mat.get());
```

Access into the dense matrix can be done using either an integer index or a mesh handle (e.g., `VertexHandle`), making it intuitive to pair matrix operations with mesh data. For the latter, make sure to use the constructor that takes `RXMesh` as first argument, i.e., 

```cpp
RXMeshStatic rx("mesh.obj");

//Allocating a N x 3 matrix (where N is the number of vertices)
DenseMatrix<float> mat(rx, rx.get_num_vertices(), 3);

//...

//Now, we can access the matrix using handles 
VertexHandle vh;
mat(vh, 0) = 1.f;
```

--- 

## **Construction**
Use one of the following constructors based on whether the matrix is mesh-aware or standalone:

Tie matrix to mesh topology (row indices will align with mesh handles):

```cpp
DenseMatrix(rx, num_rows, num_cols, location) 
```

Generic matrix not associated with mesh handles:

```cpp 
DenseMatrix(num_rows, num_cols, location)
```

User-managed memory:

```cpp
DenseMatrix(num_rows, num_cols, d_ptr, h_ptr)
```

--- 

## **Accessors**
`rows()`, `cols()` – Get matrix dimensions

`lead_dim()` – Get the leading dimension (row)

`operator()(row, col)` – Access matrix element by indices

`operator()(handle, col)` – Access matrix element by mesh handle

`col(col_id)` – Slice out a specific column as a new `DenseMatrix`

`col_data(ld_idx)` – Raw pointer to the start of a column

`data(location)` – Raw pointer to matrix data on host or device

`to_eigen()` – Return an Eigen matrix view on host memory

`bytes()` – Total bytes used in allocation

--- 

## **Modifiers**

`reset(val, location)` – Fill the matrix with a constant

`fill_random(min, max)` – Fill with random values

`reshape(new_rows, new_cols)` – Reinterpret storage without copy

`multiply(scalar)` – Multiply all elements by a scalar

`swap(other_matrix)` – Swap contents with another matrix

--- 

## ** Linear Algebra Operations**
These methods operate on the device memory and use `cuBLAS`:

`dot(other, use_conjugate)` – Dot product or element-wise inner product

`norm2()` – L2 norm of matrix

`abs_sum()` – Sum of absolute values

`abs_min()`, `abs_max()` – Element-wise extrema

`axpy(X, alpha)` – In-place Y = αX + Y

--- 

## ** Memory Management**
`move(source, target)` – Copy between host/device

`release(location)` – Free host or device memory

`copy_from(src, source_flag, target_flag)` – Deep copy with location control