# **`for_each`**

RXMesh uses **`for_each`** for two different kinds of parallel work:

| Mode | What it does | Templates |
|------|----------------|-----------|
| **Element-wise** | Visit each vertex, edge, or face with a handle only, i.e., **no** neighborhood / connectivity information | No query `Op`; optional overload dispatches on `VertexHandle`, `EdgeHandle`, or `FaceHandle` |
| **Connectivity-based** | For each seed element, traverse neighbors via an **iterator** (same neighborhood model as the `Op` enum) | `for_each<Op, blockThreads>(...)` — template args are the query `Op` and block thread count |

If you need **multiple** different `Op` operations or custom shared memory in one kernel, use [`run_kernel`](run_kernel.md).

---

## **Element-wise `for_each`**

Use this when your work uses only a **single** handle per invocation (no adjacency). Each element is a [handle](../handles.md). 

`location` is a **bitmask** of where to run, i.e., you can set `HOST` and `DEVICE` so that both code paths run. `stream` and `with_omp` apply to the device and host paths respectively (defaults: `stream = NULL`, `with_omp = true`).

```cpp
RXMeshStatic rx("mesh.obj");
auto vertex_pos = rx.get_input_vertex_coordinates();
auto vertex_color = *rx.add_vertex_attribute<float>("vColor", 3, DEVICE);

rx.for_each_vertex(
    DEVICE,
    [vertex_color, vertex_pos] __device__(const VertexHandle vh) {
        vertex_color(vh, 0) = 0.9f;
        vertex_color(vh, 1) = vertex_pos(vh, 1);
        vertex_color(vh, 2) = 0.9f;
    });
```

??? note "`for_each_vertex(location, apply, stream = NULL, with_omp = true) const`"
    Applies `apply` to each visited vertex; `apply` should take a `VertexHandle`. `location` is a bitmask of host/device execution. `stream` is the CUDA stream when the device path runs. `with_omp` controls OpenMP on the host path (default `true`); it is ignored if you do not run on `HOST`.

??? note "`for_each_edge(location, apply, stream = NULL, with_omp = true) const`"
    Same as `for_each_vertex`, but `apply` receives an `EdgeHandle`.

??? note "`for_each_face(location, apply, stream = NULL, with_omp = true) const`"
    Same as `for_each_vertex`, but `apply` receives a `FaceHandle`.

??? note "`for_each<HandleT>(location, apply, stream = NULL, with_omp = true)`"
    Dispatches to `for_each_vertex`, `for_each_edge`, or `for_each_face` when `HandleT` is `VertexHandle`, `EdgeHandle`, or `FaceHandle` respectively. Other `HandleT` types are not supported by this overload.

### **CUDA callables**

For `DEVICE` execution, `apply` must be a **device-callable** lambda (e.g. `__device__` or a `__device__ __host__` extended lambda) that capture data **by-value**. For **host-only** execution, a normal host callable is enough.

More background: [CUDA C++ Programming Guide — extended lambdas](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda).

??? info "Device launch shape (element-wise)"
    On the device path, RXMesh launches **one CUDA block per patch**, with **256** threads per block and **no** dynamic shared memory in this entry point, on the chosen `stream`.

---

## **Connectivity-based `for_each`**

These overloads launch a kernel where each thread processes one **seed** mesh element (vertex, edge, or face depending on `Op`) and exposes **neighborhood** connectivity through an [iterator](../iterators.md). The iterator type depends on `Op` (for example `Op::FV` yields vertex neighbors of each face). This is only supported on the device. 

The example computes squared edge lengths using `Op::EV`:

```cpp
RXMeshStatic rx("mesh.obj");

auto x = *rx.get_input_vertex_coordinates();
auto len = *rx.add_edge_attribute<float>("eLength", 1);

constexpr int blockSize = 256;

rx.for_each<Op::EV, blockSize>(
    [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
        Eigen::Vector3f a = x.to_eigen<3>(iter[0]);
        Eigen::Vector3f b = x.to_eigen<3>(iter[1]);
        len(eh) = (a - b).squaredNorm();
    });
```

`for_each<Op, blockThreads>` is limited to **one** query `Op` per launch. For kernels that combine multiple queries or extra shared memory, use [`run_kernel`](run_kernel.md).


??? note "`for_each<op, blockThreads>(user_lambda, oriented = false, stream = NULL) const`"
    ```cpp
    template <Op op, uint32_t blockThreads, typename LambdaT>
    void for_each(const LambdaT user_lambda,
                  const bool    oriented = false,
                  cudaStream_t  stream   = NULL) const;
    ```

    - **`op`**: compile-time query (e.g. `Op::EV`).
    - **`blockThreads`**: CUDA block size (e.g. `256`).
    - **`user_lambda`**: invoked per seed element on the device; signature `(InputHandle, OutputIterator&)`.
    - **`oriented`** (default `false`): oriented traversal where supported (e.g. `Op::VV`, `Op::VE`).
    - **`stream`**: CUDA stream for the launch (default null stream).

    RXMesh fills a [`LaunchBox<blockThreads>`](../launch_box.md) via `prepare_launch_box` for this query and launches the internal query kernel.

??? note "`for_each<op, blockThreads>(lb, user_lambda, oriented = false, stream = NULL) const`"
    ```cpp

    template <Op op, uint32_t blockThreads, typename LambdaT>
    void for_each(LaunchBox<blockThreads> lb,
                  const LambdaT           user_lambda,
                  const bool              oriented = false,
                  cudaStream_t            stream   = NULL) const;
    ```
    Same contract as above, but you supply a **precomputed** [`LaunchBox`](../launch_box.md) (using `prepare_launch_box`) to reuse launch configuration across repeated launches. The `lb` must match the same `op`, `blockThreads`, and internal query kernel.

### Supported neighbor query types {: #supported-query-types }

??? info "Vertex queries"
    | Query | Description |
    | ----- | ----------- |
    | `VV` | For vertex **V**, adjacent vertices |
    | `VE` | For vertex **V**, incident edges |
    | `VF` | For vertex **V**, incident faces |

??? info "Edge queries"
    | Query | Description |
    | ----- | ----------- |
    | `EV` | For edge **E**, incident vertices |
    | `EF` | For edge **E**, incident faces |
    | `EVDiamond` | For edge **E**, incident and opposite vertices |

??? info "Face queries"
    | Query | Description |
    | ----- | ----------- |
    | `FV` | For face **F**, incident vertices |
    | `FE` | For face **F**, incident edges |
    | `FF` | For face **F**, adjacent faces |