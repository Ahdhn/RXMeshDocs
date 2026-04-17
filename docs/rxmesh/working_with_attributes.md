# **Working with Attributes**

This section covers how to work with an attribute after allocation: query metadata, move/copy memory, and read/write values. To create or remove attributes, see [Managing Attributes](managing_attributes.md).

An RXMesh attribute is defined by:

- **Handle type**: the mesh element kind (`VertexAttribute`, `EdgeAttribute`, or `FaceAttribute`)
- **Value type**: scalar or trivially copyable type stored per component (`float`, `int`, etc.)
- **Number of components per element**: for example 3 for RGB or XYZ

```cpp
auto color = rx.add_vertex_attribute<float>("vColor", 3);  // RGB per vertex
```

The memory layout can be Struct of Arrays (SoA, default) or Array of Structs (AoS). SoA is usually preferred for GPU-friendly access patterns.

---

## **Per-Attribute Operations**

Per-attribute operations apply to all entries in the attribute (not a single element).

```cpp
auto normals = rx.add_vertex_attribute<float>("vNormal", 3);

// Query shape and metadata
auto rows = normals->rows();
auto cols = normals->cols();
auto nattr = normals->get_num_attributes();

// Manage locations
normals->move(HOST, DEVICE);
normals->reset(0.0f, DEVICE);
```

Dimension semantics:

- `rows()` is the number of mesh elements represented by the attribute
- `cols()` is the number of components per element
- `get_num_attributes()` reports the same component count as `cols()`
- `size()` is the number of mesh elements (same concept as `rows()`, not `rows() * cols()`)

??? note "`get_name() const`"
    Returns the attribute name string used when the attribute was created.

??? note "`rows() const`"
    Returns the number of mesh elements represented by this attribute.

??? note "`cols() const`"
    Returns the number of components per mesh element.

??? note "`get_num_attributes() const`"
    Returns the number of per-element components (equivalent concept to `cols()`).

??? note "`size() const`"
    Returns the number of mesh elements (equivalent concept to `rows()`).

??? note "`get_layout() const`"
    Returns the memory layout (`SoA` or `AoS`) used by the attribute.

??? note "`get_allocated() const`"
    Returns where data is currently allocated (`HOST`, `DEVICE`, or both).

??? note "`is_host_allocated() const`"
    Returns `true` if host memory is currently allocated.

??? note "`is_device_allocated() const`"
    Returns `true` if device memory is currently allocated.

??? note "`is_empty() const`"
    Returns `true` if the attribute currently has no allocated storage.

??? note "`to_matrix()`"
    Converts attribute data to a host-side dense matrix representation.

??? note "`from_matrix(mat)`"
    Loads values from a host-side dense matrix into the attribute.

---

## **Element-Wise Access**

Use these APIs to read or write one element (or one component of an element) at a time.

```cpp
VertexHandle vh = ...;

// Index-based access
float x = (*color)(10, 0);

// Handle-based access
(*color)(vh, 1) = 0.5f;

// Vector interop
auto n = color->to_glm<3>(vh);
color->from_eigen<3>(vh, eigen_n);
```

??? note "`operator()(size_t i, size_t j = 0)`"
    Accesses the `j`-th component of the `i`-th mesh element.

??? note "`operator()(HandleT handle, uint32_t attr = 0)`"
    Accesses the `attr`-th component of a specific element using its handle.

??? note "`to_glm<N>(handle)` / `from_glm<N>(handle, value)`"
    Converts between attribute components and GLM vector types for one element.

??? note "`to_eigen<N>(handle)` / `from_eigen<N>(handle, value)`"
    Converts between attribute components and Eigen vectors for one element.

`N` must match the per-element component count (for example, `N = 3` for RGB or XYZ).

---

## **Memory and Lifecycle**

These APIs control where data lives and how it is copied or cleared.

```cpp
// Reset on device asynchronously on stream s
cudaStream_t s = ...;
attr->reset(0.0f, DEVICE, s);

// Ensure host copy exists
attr->move(DEVICE, HOST, s);

// Copy from another attribute
dst->copy_from(src, LOCATION_ALL, LOCATION_ALL, s);
```

??? note "`reset(value, location = LOCATION_ALL, stream = NULL)`"
    Fills all entries with `value` in the requested `location` (`HOST`, `DEVICE`, or both). If a CUDA stream is provided, device work is enqueued on that stream.

??? note "`move(source, target, stream = NULL)`"
    Moves data from `source` to `target` location. If target storage is missing, it is allocated first. When a stream is provided, device transfers run on that stream.

??? note "`copy_from(source, source_flag = LOCATION_ALL, dst_flag = LOCATION_ALL, stream = NULL)`"
    Deep-copies from `source` into this attribute, filtered by source and destination location flags. With `LOCATION_ALL`, RXMesh copies host-to-host and device-to-device where available; stream controls device-side copy scheduling.

??? note "`release(location = LOCATION_ALL)`"
    Releases allocated storage in the requested location(s). Use `LOCATION_ALL` to free both host and device allocations.

---

## **Practical Notes**

- Prefer handle-based access inside mesh algorithms, and index-based access for linear host-side loops.
- Use `to_matrix()` / `from_matrix()` for batch host workflows (analysis, serialization, or interop).
- Pick SoA for most GPU kernels; consider AoS only when your access pattern benefits from packed per-element structs.
- Some low-level overloads and patch-local indexing utilities exist for advanced/internal usage; most users should rely on the APIs documented here.