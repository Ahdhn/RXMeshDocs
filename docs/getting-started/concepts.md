# **Key Concepts**

This page introduces the core ideas behind RXMesh. Understanding these concepts will make the rest of the documentation much easier to follow.

---

## **Patches**

RXMesh does not store the mesh as a single monolithic array. Instead, during construction, the input mesh is **partitioned into patches**—small, local subsets of the mesh that fit into GPU shared memory. Each patch contains a group of vertices, edges, and faces along with their local connectivity.

This patch-based design is central to RXMesh's performance. By keeping patches small enough to reside in shared memory, neighborhood queries can be answered without accessing global memory. Users generally do not interact with patches directly—RXMesh manages partitioning, locality, and load balancing internally. However, patches influence how [indexing](../rxmesh/indexing.md) works and explain why RXMesh uses **handles** rather than raw integer indices.

---

## **Handles**

Every mesh element (vertex, edge, or face) is identified by a **handle**—a lightweight 64-bit identifier that encodes both the **patch ID** and the element's **local index** within that patch.

RXMesh defines three handle types:

| Handle Type    | Represents |
|----------------|------------|
| `VertexHandle` | A vertex   |
| `EdgeHandle`   | An edge    |
| `FaceHandle`   | A face     |

Handles are the primary way to refer to mesh elements throughout the API. You use them to read and write [attributes](../rxmesh/attributes.md), and they are passed into lambdas by [`for_each`](../rxmesh/for_each.md) and [query kernels](../rxmesh/run_query_kernel.md).

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

- A 3D position per vertex (`VertexAttribute<float>` with 3 components)
- A scalar curvature per edge (`EdgeAttribute<float>` with 1 component)
- A label per face (`FaceAttribute<int>` with 1 component)

Attributes are **strongly typed**—a `VertexAttribute` can only be indexed with a `VertexHandle`, preventing accidental misuse.

Attributes can reside on the **host** (CPU), **device** (GPU), or **both**, and you explicitly control data movement between them:

```cpp
auto color = *rx.add_vertex_attribute<float>("vColor", 3);

// Compute on GPU...

color.move(DEVICE, HOST);  // bring results back to CPU
```

Attributes are covered in detail under [Managing Attributes](../rxmesh/attributes_management.md) (creating, checking, removing) and [Working with Attributes](../rxmesh/attributes.md) (accessing values, memory movement, Eigen/GLM interop).

---

## **Operations: `for_each` vs Queries**

RXMesh provides two main ways to run computations over the mesh:

### Element-Wise Operations (`for_each`)

Use `for_each` when your computation depends only on a single element, with no need to access neighbors. The lambda receives one handle at a time:

```cpp
rx.for_each_vertex(DEVICE, [color] __device__(const VertexHandle vh) {
    color(vh, 0) = 0.5f;
});
```

This runs a parallel loop over all vertices (or edges, or faces). See [`for_each`](../rxmesh/for_each.md).

### Query Kernels (`run_query_kernel`)

Use a query kernel when you need to access **neighboring elements**—for example, the vertices of a face or the one-ring neighbors of a vertex. Queries are specified using the `Op` enum:

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

A query kernel receives the input element's handle and an **iterator** over the output elements:

```cpp
rx.run_query_kernel<Op::FV, 256>(
    [=] __device__(FaceHandle fh, VertexIterator& fv) mutable {
        // fv[0], fv[1], fv[2] are the three vertex handles
        auto p0 = coords.to_glm<3>(fv[0]);
        auto p1 = coords.to_glm<3>(fv[1]);
        auto p2 = coords.to_glm<3>(fv[2]);
    });
```

See [Query Kernels](../rxmesh/run_query_kernel.md) for the full API and query table. For advanced use cases where you need to combine multiple queries or use shared memory, see [Custom Kernels](../rxmesh/run_kernel.md).

---

## **Host and Device**

RXMesh is designed for GPU execution. The two key locations for data are:

- **`HOST`** — CPU memory. Used for I/O, visualization, debugging.
- **`DEVICE`** — GPU memory. Used for computation.

Most computation in RXMesh happens on the device. Lambdas passed to `for_each` or query kernels must be annotated with `__device__` and must capture data by value (not by reference). After computation, results are typically moved back to the host for output or visualization:

```cpp
attribute.move(DEVICE, HOST);
```

---

## **Putting It Together**

A typical RXMesh workflow looks like this:

1. **Initialize** RXMesh and load a mesh → `RXMeshStatic rx("mesh.obj")`
2. **Create attributes** to store data on mesh elements → `rx.add_vertex_attribute<float>(...)`
3. **Run computations** using `for_each` or query kernels
4. **Move results** to the host if needed → `attr.move(DEVICE, HOST)`
5. **Visualize or export** → Polyscope, OBJ, VTK

For the complete API, continue to [Static Mesh Processing](../rxmesh/static.md).
