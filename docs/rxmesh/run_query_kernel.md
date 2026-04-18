# **`run_query_kernel` **

`run_query_kernel` is used to launch a device kernel that operates on a **mesh element and its local neighborhood**. It takes a query type (e.g., `Op::EV`), which specifies what neighborhood relation to access, e.g., "incident vertices of an edge" or "adjacent faces of a face". This is the *simplest* entry point in RXMesh for neighborhood-based computations. The neighborhood is exposed through an [Iterator](iterators.md) whose type depends on the query's output element, e.g., a vertex one-ring query (`VV`) yields a `VertexIterator`.

The example below computes the squared rest length for each edge using the `EV` query:

```cpp
RXMeshStatic rx("mesh.obj");

auto x = *rx.get_input_vertex_coordinates(); // vertex positions
auto len = *rx.add_edge_attribute<T>("eLength", 1);

constexpr int blockSize = 256;

rx.run_query_kernel<Op::EV, blockSize>(
    [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
        Eigen::Vector3<T> a = x.to_eigen<3>(iter[0]);
        Eigen::Vector3<T> b = x.to_eigen<3>(iter[1]);

        len(eh) = (a - b).squaredNorm();
});
```

---
## **Supported Query Types**

**Vertex Queries** 


| Query       | Description                                    |
| ----------- | ---------------------------------------------- |
| `VV`  🟢 ➔ 🟢 | For vertex **V**, return its adjacent vertices |
| `VE`  🟢 ➔ ➖ | For vertex **V**, return its incident edges    |
| `VF`  🟢 ➔ 🔺 | For vertex **V**, return its incident faces    |

**Edge Queries** 

| Query               | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `EV`   ➖ ➔ 🟢        | For edge **E**, return its incident vertices              |
| `EF`   ➖ ➔ 🔺        | For edge **E**, return its incident faces                 |
| `EVDiamond`   ➖ ➔ 🟢 | For edge **E**, return its incident and opposite vertices |

**Face Queries** 

| Query     | Description                                  |
| --------- | -------------------------------------------- |
| `FV` 🔺➔ 🟢 | For face **F**, return its incident vertices |
| `FE` 🔺➔ ➖ | For face **F**, return its incident edges    |
| `FF` 🔺➔ 🔺 | For face **F**, return its adjacent faces    |


---

## **Usage**

`run_query_kernel` is limited to **one** query `Op` per launch. For kernels that combine multiple queries or custom shared memory, use the more-generic [`run_kernel`](run_kernel.md) API.

Both overloads are `const` member functions on `RXMeshStatic`. They are parameterized by the query `Op` and the CUDA block size `blockThreads`, and take a **device-callable** lambda whose first argument is the query’s **input handle** and whose second argument is an **output iterator** over neighbors (the concrete handle and iterator types depend on `Op`; see [Iterators](iterators.md)). Capture attributes and mesh data **by value** in the lambda.

??? note "`run_query_kernel<op, blockThreads>(user_lambda, oriented = false, stream = NULL)`"    
    - **`op`**: compile-time query (e.g. `Op::EV`). Only one `Op` per call.  
    - **`blockThreads`**: threads per CUDA block (e.g. `256`). 
    - **`user_lambda`**: invoked on the device for each input element; signature is `(InputHandle, OutputIterator&)` where types follow from `op`.
    - **`oriented`** (optional, default `false`): when `true`, enables oriented traversal for queries that support it (e.g. `Op::VV`, `Op::VE`).
    - **`stream`** (optional): CUDA stream for the launch; default is the null stream.
    
    Internally, RXMesh builds a [`LaunchBox<blockThreads>`](launch_box.md) via `prepare_launch_box` for this single query, then launches the query kernel.

??? note "`run_query_kernel<op, blockThreads>(lb, user_lambda, oriented = false, stream = NULL)`"    
    Same template parameters and lambda contract as the overload above but here the user supplies a **pre-computed** [`LaunchBox<blockThreads>`](launch_box.md). Use this when you already called `prepare_launch_box` (for example to reuse the same launch configuration across repeated launches). The `lb` must be consistent with the same `op` and block size as the internal query kernel.
