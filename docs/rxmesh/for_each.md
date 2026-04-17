# **`for_each`**

RXMesh provides `for_each` functions that apply a user-defined callable to mesh elements of a given type—vertices, edges, or faces. Each element is identified by a [handle](handles.md). These traversals are embarrassingly parallel and do __not__ rely on connectivity or neighbor traversal. When your logic needs adjacent elements (for example, a face’s vertices), use [`run_query_kernel`](run_query_kernel.md) instead.

```cpp
RXMeshStatic rx("mesh.obj");
auto vertex_pos = rx.get_input_vertex_coordinates();
auto vertex_color = rx.add_vertex_attribute<float>("vColor", 3, DEVICE);

rx.for_each_vertex(
    DEVICE,
    [vertex_color, vertex_pos] __device__(const VertexHandle vh) {
        vertex_color(vh, 0) = 0.9f;
        vertex_color(vh, 1) = vertex_pos(vh, 1);
        vertex_color(vh, 2) = 0.9f;
    });
```

??? note "`for_each_vertex(location, apply, stream = NULL, with_omp = true) const`"    
    Applies `apply` to each visited vertex; `apply` should take a `VertexHandle`. `location` is a bitmask of host/device execution sites. `stream` is the CUDA stream when device execution is requested. `with_omp` enables OpenMP for host execution (default `true`); it is ignored when only the device path runs.

??? note "`for_each_edge(location, apply, stream = NULL, with_omp = true) const`"    
    Same as `for_each_vertex`, but `apply` receives an `EdgeHandle`.

??? note "`for_each_face(location, apply, stream = NULL, with_omp = true) const`"    
    Same as `for_each_vertex`, but `apply` receives a `FaceHandle`.

??? note "`for_each<HandleT>(location, apply, stream = NULL, with_omp = true)`"    
    Dispatches to `for_each_vertex`, `for_each_edge`, or `for_each_face` according to `HandleT`. `HandleT` must be `VertexHandle`, `EdgeHandle`, or `FaceHandle`.

### **CUDA lambdas**

For **`DEVICE`** execution, `apply` must satisfy RXMesh’s compile-time checks for a device callable (for example a `__device__` or `__host__ __device__` extended lambda, as recognized by RXMesh’s traits). Captures used on the device must be valid in device code (typically **by-value** captures).

For **`HOST`**-only execution, an ordinary host callable is sufficient.

More background: [CUDA C++ Programming Guide — extended lambdas](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda).

??? info "Device launch shape"
    On the device path, `RXMeshStatic` launches one block per patch (`grid.x = num_patches`) with **256** threads per block, **0** bytes of dynamic shared memory, on `stream`.