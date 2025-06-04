# **Initialization**

`RXMeshStatic` can be constructed in two ways, depending on whether the input mesh is stored in a file or directly provided as face connectivity in memory.

---

## **From an `.obj` File**

```cpp
RXMeshStatic(const std::string obj_file_path,
             const std::string patcher_file = "",
             const uint32_t    patch_size   = 512);
```

This constructor takes a path to a triangle mesh stored as a `.obj` file and builds a patch-based internal representation optimized for GPU execution.

- `obj_file_path`: Path to the input .obj mesh file.

- `patcher_file`: Optional. If provided, RXMesh will use the patch layout stored in this file instead of generating new patches.

- `patch_size`: Controls the target average number of faces per patch. The default is 512.

### Patch-Based Construction and Determinism
RXMesh uses a patch-based decomposition of the mesh to improve data locality and enable efficient parallel processing. By default, this patching process is run at every construction and includes randomness (e.g., random starting points or ordering) which can lead to non-deterministic patch layouts. This does not affect correctness but can make debugging more difficult.

To make the patching deterministic, users can supply a precomputed patch layout via the `patcher_file` argument. This file stores the result of a previous patching run and allows RXMesh to reuse the same layout. To generate such a file, call:

```c++
RXMeshStatic rx("mesh.obj");
rx.save("patches.rx");
```
You can then pass `patches.rx` as the `patcher_file` argument in subsequent constructions.

--- 

## **From memory**
```cpp
RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
             const std::string patcher_file = "",
             const uint32_t    patch_size   = 512);
```

This constructor builds the mesh from an in-memory face list, where each face is represented by a vector of three vertex indices.

- `fv`: Face list. Each entry contains the three vertex indices for one triangle.

- `patcher_file`, `patch_size`: Same as above.

Note that this constructor only defines the mesh topology (i.e., connectivity). To provide vertex positions, call:

```cpp
rx.add_vertex_coordinates(std::vector<std::vector<float>>& vertex_coords);
```
Here, `vertex_coords[i]` should be a 3-element vector containing the 3D position of vertex `i`.

Both constructors create an internal mesh representation suitable for attribute management, kernel launches, and connectivity queries. The choice between them depends on whether the input mesh is available as a file or already loaded into memory.
