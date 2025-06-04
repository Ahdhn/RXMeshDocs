# **`for_each` **

RXMesh provides four `for_each` functions that apply a user-defined lambda function over all mesh elements of a given type: vertices, edges, or faces. These operations are embarrassingly parallel—they do not rely on connectivity or neighbor traversal.

You can run them on the host (parallelized with OpenMP) or the device (as CUDA kernels). The lambda must accept a mesh element [handle](handles.md) (e.g., VertexHandle) and be annotated with `__device__` if executed on the GPU. For GPU execution, all data captures must be by value. More about lambda function in CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda).

---
## Functions
- `for_each_vertex(location, lambda, stream, with_omp)`

- `for_each_edge(location, lambda, stream, with_omp)`

- `for_each_face(location, lambda, stream, with_omp)`

- `for_each<HandleT>(location, lambda, stream, with_omp)` — generic version

---

## Usage

- `location`: one of `HOST` or `DEVICE`
- `lambda`: must match the handle type (e.g., `const FaceHandle fh`).
- `stream`: only used when running on the GPU.
- `with_omp`: only relevant when executing on the host; defaults to true.

---
## Example: Per-Vertex Coloring
```c++
RXMeshStatic rx("mesh.obj");
auto vertex_pos = rx.get_input_vertex_coordinates();
auto vertex_color = rx.add_vertex_attribute<float>("vColor", 3, DEVICE);

rx.for_each_vertex(
    DEVICE,
    [vertex_color, vertex_pos] __device__ (const VertexHandle vh) {
        vertex_color(vh, 0) = 0.9;
        vertex_color(vh, 1) = vertex_pos(vh, 1);
        vertex_color(vh, 2) = 0.9;
});
```

---
## When to Use

Use `for_each` when:

- You want to apply a function per element.
- The function does **not** require access to adjacent elements.
- The logic can be done independently per vertex, edge, or face.

If you do need access to neighbor elements (e.g., face's vertices), use `run_query_kernel`.