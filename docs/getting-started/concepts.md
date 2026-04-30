# **Key Concepts**

This page introduces the core ideas behind RXMesh system. Understanding these concepts will make the rest of the documentation much easier to follow.

---

## **Patches**

RXMesh does not store the mesh as a single monolithic array. Instead, during construction, the input mesh is **partitioned into patches**. A patch is a small subset connected mesh faces that fit into GPU shared memory. Each patch contains a group of vertices, edges, and faces along with their local connectivity.

This patch-based design is central to RXMesh's performance. By keeping patches small enough to reside in shared memory, neighborhood queries can be answered without accessing global memory. Users generally do not interact with patches directly, i.e., RXMesh manages partitioning, locality, and load balancing internally. However, patches influence how [indexing](../rxmesh/indexing.md) works and explain why RXMesh uses **handles** rather than raw integer indices.

---

## **Handles**

Every mesh element (vertex, edge, or face) is identified by a **handle**, i.e., a 64-bit identifier that encodes both the **patch ID** and the element's **local index** within that patch.

RXMesh defines three handle types:

| Handle Type    | Represents |
|----------------|------------|
| `VertexHandle` | A vertex   |
| `EdgeHandle`   | An edge    |
| `FaceHandle`   | A face     |

Handles are the primary way to refer to mesh elements throughout RXMesh. You use handles to identify elements when you read or write [attributes](../rxmesh/working_with_attributes.md). Handles are used also as the signature of the lambda function passed to [`for_each`](../rxmesh/for_each.md) operations.

```cpp
rx.for_each_vertex(
    DEVICE, [vertex_color] __device__(const VertexHandle vh) {
        vertex_color(vh, 0) = 1.0f;
    });
```

Handles are **not** contiguous integers. If you need a flat index (e.g., for exporting or interfacing with external libraries), see [Indexing](../rxmesh/indexing.md).

For the full handle API, see the [Handles](../rxmesh/handles.md) reference.

---

## **Attributes**

An **attribute** is a typed array of values attached to mesh elements. For example:

- A per-vertex 3D position (`VertexAttribute<float>` with 3 components)
- A per-edge scalar edge length (`EdgeAttribute<float>` with 1 component)
- A per-face scalar face area (`FaceAttribute<int>` with 1 component)

Attributes are **strongly typed**, i.e., a `VertexAttribute` can only be indexed with a `VertexHandle`, preventing accidental misuse.

Attributes can reside on the **host** (CPU), **device** (GPU), or **both**, and you explicitly control data movement between them:

```cpp
auto color = *rx.add_vertex_attribute<float>("vColor", 3);

// Compute on GPU...

color.move(DEVICE, HOST);  // bring results back to CPU
```

Attributes are covered in detail under [Managing Attributes](../rxmesh/managing_attributes.md) (creating, checking, removing) and [Working with Attributes](../rxmesh/working_with_attributes.md) (accessing values, memory movement, Eigen/GLM interop).

---

## **`for_each` Operations**

RXMesh provides two main ways to run computations over the mesh:

### Element-wise Operations

Use `for_each` when your computation depends only on a single element, with no need to access neighbors. The lambda receives one handle at a time:

```cpp
rx.for_each_vertex(DEVICE, [color] __device__(const VertexHandle vh) {
    color(vh, 0) = 0.5f;
});
```

This runs a parallel loop over all vertices (or edges, or faces). See [`for_each`](../rxmesh/for_each.md).

### Connectivity-based Operations

Use `for_each<Op, blockThreads>` when you need to access **neighboring elements**, e.g., the vertices of a face or the one-ring neighbors of a vertex. Neighbor queries are specified using the `Op` enum:

| Op      | Meaning                                |
|---------|----------------------------------------|
| `Op::VV` | For each vertex, its adjacent vertices |
| `Op::VE` | For each vertex, its incident edges    |
| `Op::VF` | For each vertex, its incident faces    |
| `Op::EV` | For each edge, its two vertices        |
| `Op::EF` | For each edge, its incident faces      |
| `Op::FV` | For each face, its three vertices      |
| `Op::FE` | For each face, its three edges         |
| `Op::FF` | For each face, its adjacent faces      |
| `Op::EVDiamond` | For each edge, its incident and opposite vertices |

The lambda receives the input element's handle and an **iterator** over the output elements:

```cpp
rx.for_each<Op::FV, 256>(
    [=] __device__(FaceHandle fh, VertexIterator& fv) {
        // fv[0], fv[1], fv[2] are the three vertex handles
        auto p0 = coords.to_glm<3>(fv[0]);
        auto p1 = coords.to_glm<3>(fv[1]);
        auto p2 = coords.to_glm<3>(fv[2]);
    });
```

See [Connectivity `for_each`](../rxmesh/for_each.md#connectivity-for_each-query-op) for the full API and [supported query types](../rxmesh/for_each.md#supported-query-types). For advanced use cases where you need to combine multiple queries or use shared memory, see [Custom Kernels](../rxmesh/run_kernel.md).

---

## **Host and Device**

RXMesh is designed for GPU execution. The two key locations for data are:

- **`HOST`**: CPU memory. Used for I/O, visualization, debugging.
- **`DEVICE`**: GPU memory. Used for computation.

Most computation in RXMesh happens on the device. Lambdas passed to `for_each` (element-wise or query overloads) must be annotated with `__device__` and must capture data by value (not by reference). After computation, results are typically moved back to the host for output or visualization:

```cpp
attribute.move(DEVICE, HOST);
```

---

## **Putting It Together**

The most basic workflow in RXMesh looks like this:

1. **Initialize** RXMesh and load a mesh → `RXMeshStatic rx("mesh.obj")`
2. **Create attributes** to store data on mesh elements → `rx.add_vertex_attribute<float>(...)`
3. **Run computations** using element-wise `for_each` or `for_each<Op, blockThreads>`.  
4. **Move results** to the host if needed → `attr.move(DEVICE, HOST)`
5. **Visualize or export** → Polyscope

For the complete API, continue to [Static Mesh Processing](../rxmesh/static.md).  Step 3 can also become more advanced, e.g., solving linear systems with [Matrices & Solvers](../rxmesh/solvers.md), performing [dynamic mesh operations](../rxmesh/dyn/dynamic.md) that change connectivity, or computing derivatives with [automatic differentiation](../rxmesh/diff/ad.md) for non-linear optimization.
