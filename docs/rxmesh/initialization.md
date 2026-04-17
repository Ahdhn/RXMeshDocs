# **Initialization**

`RXMeshStatic` can be constructed in three ways, depending on whether the input mesh is stored in one `.obj` file, multiple `.obj` files, or directly provided as face connectivity and vertex coordinates in memory.

---

## **From an `.obj` File**

This constructor takes a path to a triangle mesh stored as a `.obj` file and builds the internal data structure optimized for GPU execution.

??? note "`RXMeshStatic(const std::string file_path, const std::string patcher_file = "", const uint32_t patch_size = 512, const float capacity_factor = 1.0, const float patch_alloc_factor = 1.0, const float lp_hashtable_load_factor = 0.8)`"
    - `file_path`: Path to the input `.obj` file.
    - `patcher_file` (optional): Path to a previously saved patch layout (`rx.save(...)`) to restore deterministic patch assignment.
    - `patch_size` (optional): Target average number of faces per patch (default: `512`).
  
    The following parameters are only relevant to `RXMeshDynamic` which inherits from `RXMeshStatic`

    - `capacity_factor` (optional): Extra per-patch storage headroom before slicing/splitting is needed (default: `1.0`). 
    - `patch_alloc_factor` (optional): Extra global patch allocation headroom beyond the initial k-means patch count (default: `1.0`).
    - `lp_hashtable_load_factor` (optional): Load factor for the hashtable used for mapping non-owned elements to their owner `(patch_id, local_id)` (default: `0.8`).

### Patch-Based Construction and Determinism
RXMesh uses a patch-based decomposition of the mesh to improve data locality and enable efficient parallel processing. By default, this patching process is run at every construction and includes randomness (e.g., random starting points or ordering) which can lead to non-deterministic patch layouts. This does not affect correctness but can make debugging more difficult.

To make the patching deterministic, users can supply a precomputed patch layout via the `patcher_file` argument. This file stores the result of a previous patching run and allows RXMesh to reuse the same layout. To generate such a file, call:

```c++
RXMeshStatic rx("mesh.obj");
rx.save("patches.rx");
```
You can then pass `patches.rx` as the `patcher_file` argument in subsequent constructions.

--- 

## **From multiple `.obj` files**

This constructor loads and merges multiple `.obj` files into one RXMesh instance.

??? note "`RXMeshStatic(const std::vector<std::string> files_path, const uint32_t patch_size = 512)`"
    - `files_path`: List of input `.obj` file paths.
    - `patch_size` (optional): Target average number of faces per patch (default: `512`).

When using multiple input meshes, RXMesh also tracks region information (one region index per input mesh) which you can access through built-in region-label attributes described in [Managing Attributes](managing_attributes.md#built-in-attributes).

---

## **From memory**

This constructor builds the mesh from an in-memory face list where each face is represented by a vector of three vertex indices.

??? note "`RXMeshStatic(std::vector<std::vector<uint32_t>>& fv, const std::string patcher_file = \"\", const uint32_t patch_size = 512, const float capacity_factor = 1.0, const float patch_alloc_factor = 1.0, const float lp_hashtable_load_factor = 0.8)`"
    - `fv`: Face list (triangle connectivity). Each entry is a vector of three vertex indices.
    - `patcher_file` (optional): Path to a previously saved patch layout for deterministic patch assignment.
    - `patch_size` (optional): Target average number of faces per patch (default: `512`).
    
    The following parameters are only relevant to `RXMeshDynamic` which inherits from `RXMeshStatic`

    - `capacity_factor` (optional): Extra per-patch storage headroom before slicing/splitting is needed (default: `1.0`).
    - `patch_alloc_factor` (optional): Extra global patch allocation headroom beyond the initial k-means patch count (default: `1.0`).
    - `lp_hashtable_load_factor` (optional): Load factor for the hashtable used for mapping non-owned elements to their owner `(patch_id, local_id)` (default: `0.8`).

Note that this constructor only defines the mesh topology (i.e., connectivity). To provide vertex positions, call:

```cpp
rx.add_vertex_coordinates(std::vector<std::vector<float>>& vertex_coords);
```
Here, `vertex_coords[i]` should be a 3-element vector containing the 3D position of vertex `i`.
