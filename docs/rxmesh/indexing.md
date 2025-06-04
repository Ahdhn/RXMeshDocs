# ** Indexing **

At a high level, RXMesh operates by first partitioning the input mesh into a set of patches. Each patch contains a local subset of the mesh's vertices, edges, and faces. Then, we launch work/operations on these patches and this patch-based design enables efficient parallel execution on the GPU by leveraging locality and compact data access.

To represent a mesh element (vertex, edge, or face) in this structure, RXMesh uses a [handle](handles.md), which is a lightweight tuple of the form (`patch_id`, `local_index`). This means each handle refers to a specific element within a specific patch. Handles are used throughout the library to interact with mesh elements—for example, when launching per-element computations with `for_each` or `run_query_kernel`.

While handles are ideal for internal operations, users may need more interpretable or flattened indexing schemes for debugging, exporting data, or integrating with external tools. RXMesh provides two types of user-facing indices:

---
## ** Linear ID **

Linear IDs are compact indices assigned to each mesh element by traversing all patches sequentially. For example:

- Patch 0 has 10 vertices, their linear IDs are 0 through 9.

- Patch 1 has 5 vertices, their linear IDs are 10 through 14.

- Patch 2 continues the sequence, and so on.

This gives a continuous/flat indexing of mesh elements across the entire mesh. Linear IDs can be useful for outputting mesh attributes as flat arrays or for mapping internal RXMesh representations to external data structures.

You can access the linear ID of a mesh element as follows:

- On the host:
```cpp
RXMeshStatic::linear_id(handle)
```

- On the device (inside a kernel):
```cpp
context.linear_id(handle)
```

---

## ** `map_to_global` **


Global indices are the original indices assigned to vertices and faces in the input `.obj` file. These indices are preserved to allow traceability to the original mesh topology.

This is useful for:

- Validating output data against known reference meshes.
- Debugging: mapping a computed value back to the original mesh element.


Global indices can only be queried for vertices and faces on the host using:

```cpp
RXMeshStatic::map_to_global(handle)
```

Note: Edges do not exist explicitly in the `.obj` file and hence do not have global indices.