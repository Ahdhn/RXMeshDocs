# ** Sparse Matrices ** 

The `SparseMatrix` class represents a sparse matrix in compressed sparse row (CSR) format. Each matrix entry corresponds to a mesh-based connectivity pattern defined by a query operation (e.g., `Op::VV` for vertex adjacency). Internally, it stores row pointers, column indices, and values on both the host and the device.

The matrix can be accessed using standard row-column indexing or using mesh handles. It supports both library-managed and user-managed memory.

---

## **Construction**

From a mesh and query:
```cpp
SparseMatrix(rx, Op::VV)
```

From user-managed memory:
```cpp
SparseMatrix(num_rows, num_cols, nnz, d_row_ptr, d_col_idx, d_val, h_row_ptr, h_col_idx, h_val)
```

When constructed using a mesh query (e.g., Op::VV), the matrix will has shape (`num_vertices × num_vertices`) for VV, it will be populated such that `(i, j)` is non-zero if the vertices `i` and `j` are neighbors. This also allows access via `SparseMatrix(vh1, vh2)` where `vh1` and `vh2` are `VertexHandle`.

---

## **Accessors**

`rows()`, `cols()` – Number of rows and columns

`non_zeros()` – Total number of non-zero entries

`non_zeros(row_id)` – Number of non-zeros in a specific row

`col_id(row_id, nnz_index)` – Get the column index for a specific row/nnz

`get_val_at(idx)` – Get value by linear nnz index

`row_ptr()`, `col_idx()`, `val_ptr()` – Pointers to raw CSR arrays

`get_row_id(handle` – Map a handle to the row ID

Access matrix entries:

- `operator()(row, col)` – Access via integer indices

- `operator()(RowHandle, ColHandle)` – Access via mesh handles


---

## **Modifiers**

`reset(value, location)` – Set all entries to a constant

`for_each(lambda)` – Visit all entries (host-only)

`transpose()` – Return transposed matrix

`move(source, target)` – Move between host/device

`to_file("out.txt")` – Export matrix to text (MATLAB format)


---

## **Matrix Operations**

### Multiplication with Dense Matrix

```cpp 
multiply(B, C, transpose_A, transpose_B, alpha, beta)
```

Performs the generalized sparse-dense matrix multiplication:

```cpp
C = alpha * op(A) * op(B) + beta * C
```

- `A` is the sparse matrix (this object)

- `B` is a dense matrix

- `C` is the output dense matrix

- `alpha` and `beta` are scalar coefficients

- `op(X)` denotes optionally transposing or conjugate-transposing the matrix (controlled via `transpose_A` / `transpose_B`)

This operation uses `cuSPARSE` and supports buffer reuse: you can call `allocate_multiply_buffer()` ahead of time to preallocate temporary workspace. This avoids allocating memory each time and makes timing results cleaner.

### Multiplication with Dense Vector

```cpp
multiply(input_ptr, result_ptr)
```

Performs a standard sparse matrix-vector multiplication:

```cpp 
Y = A * X
```

- `A` is the sparse matrix

- `X` and `Y` are raw pointers to the input and output vectors (on device)

This version is optimized for 1D vector inputs and is also backed by `cuSPARSE`.

### Column-Wise Multiplication

```cpp
multiply_cw(B, C)
```

Computes:

```cpp
C = A * B
```

but does so column-wise, treating each column of `B` as a separate vector. Internally, this calls the sparse matrix-vector multiply in a loop and is useful when `B` is tall and skinny (e.g., features stored per vertex).

This version trades peak performance for flexibility and is suitable when you cannot reshape or fuse operations into a batched matrix-matrix product.

---

## **Interop**

`to_eigen()` – Zero-copy conversion to Eigen sparse matrix

`to_eigen_copy()` – Copy-based conversion to Eigen


