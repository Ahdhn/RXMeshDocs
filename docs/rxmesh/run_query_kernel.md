# **`run_query_kernel` **

`run_query_kernel` is used to launch a device kernel that operates on a **mesh element and its local neighborhood**. It takes a query type (e.g., `Op::EV`), which specifies what neighborhood relation to access—such as "incident vertices of an edge" or "adjacent faces of a face". This is the *simplest* entry point in RXMesh for neighborhood-based computations. The neighborhood is exposed through an [Iterator](iterators.md) whose type depends on the query's output element—for example, a vertex one-ring query (`VV`) yields a `VertexIterator`.

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

`run_query_kernel` comes in two variants:

- A simpler overload that automatically prepares the [*LaunchBox*](launch_box.md).
- A variant that accepts a user-supplied [*LaunchBox*](launch_box.md) if you want to manually set up the kernel configuration.

Both require:

- The **query type** as a template parameter (e.g., `Op::EV`)
- The **CUDA block size**
- A **lambda function** that accepts a mesh element and an iterator over its neighbors.

Lambdas must be annotated `__device__` and capture variables by value.

---

## **Example: Computing Edge Length**

This example computes the squared rest length for each edge using the `EV` query.

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

> See [Managing Attributes](managing_attributes.md) to learn how to create and manipulate attributes like `len`.

