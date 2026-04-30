# **Working with Attributes**

This section focuses on what you usually do after creating an attribute, i.e., read/write values, reset data, move data between host/device, and copy from another attribute. To create or remove attributes, see [Managing Attributes](managing_attributes.md).

An RXMesh attribute is defined by:

- **Handle type**: the mesh element kind (`VertexAttribute`, `EdgeAttribute`, or `FaceAttribute`)
- **Value type**: scalar or trivially copyable type stored per component (`float`, `int`, etc.)
- **Number of components per element**: for example 3 for RGB or XYZ

```cpp
auto color = *rx.add_vertex_attribute<float>("vColor", 3);  // RGB per vertex
```

The memory layout can be Struct of Arrays (SoA, default) or Array of Structs (AoS). SoA is usually preferred for GPU-friendly access patterns.

---

## **Core Workflow**

Most day-to-day code touches four APIs, i.e., `operator()` for per-element access, `reset(...)` to overwrite all entries, `move(...)` to transfer/ensure location, and `copy_from(...)` to copy values from another attribute.

Read or write one component with `operator()`:

```cpp
VertexHandle vh = ...;

float red = color(vh, 0);   // Handle-based read
color(10, 1) = 0.25f;       // Index-based write (element 10, component 1)
```

??? note "`operator()(HandleT handle, uint32_t attr = 0)`"
    Accesses the `attr`-th component of one element via its mesh handle.

??? note "`operator()(size_t i, size_t j = 0)`"
    Accesses the `j`-th component of the `i`-th mesh element by index.

Reset all values:

```cpp
cudaStream_t s = ...;
color->reset(0.0f, DEVICE, s);   // Asynchronous device fill on stream s
color->reset(1.0f, HOST);        // Synchronous host fill
```

??? note "`reset(value, location = LOCATION_ALL, stream = NULL)`"
    Fills all entries with `value` in `HOST`, `DEVICE`, or both. If a CUDA stream is provided, device work is enqueued on that stream.

Move data to from device to host (or vice verse):

```cpp
cudaStream_t s = ...;
color->move(HOST, DEVICE, s);  // Upload for a GPU kernel
color->move(DEVICE, HOST, s);  // Bring results back to host
```

??? note "`move(source, target, stream = NULL)`"
    Moves data from `source` location to `target` location. Missing target storage is allocated first.

Copy values from a compatible attribute:

```cpp
auto src = rx.add_vertex_attribute<float>("vColor_src", 3);
auto dst = rx.add_vertex_attribute<float>("vColor_dst", 3);

cudaStream_t s = ...;
dst->copy_from(src, LOCATION_ALL, LOCATION_ALL, s);
```

??? note "`copy_from(source, source_flag = LOCATION_ALL, dst_flag = LOCATION_ALL, stream = NULL)`"
    Deep-copies from `source` into this attribute, with optional source/destination location filtering.

---

## **Element-Wise Access Patterns**

Use **index-based access** for direct host-side loops over compact indices. Use **handle-based access** inside mesh algorithms and kernels where handles are the natural key.

```cpp
// Mesh-algorithm style
VertexHandle vh = ...;
color(vh, 0) = 0.8f;

// Host-side index loop style
for (size_t i = 0; i < color->size(); ++i) {
    color(i, 2) = 1.0f;
}
```

Because `size()` reports element count, loops over `i = 0..size()-1` iterate mesh elements (not component slots).

---

## **Vector / Linear-Algebra Interop**

When you need local geometry math, convert per-element components to vectors, compute, then write back.

Example: compute one face normal from three vertex positions using Eigen:

```cpp
auto position = *rx.get_input_vertex_coordinates();  // VertexAttribute<rx_coord_t>, 3 comps
auto face_normal = *rx.add_face_attribute<float>("fNormal", 3);

//inside GPU kernel 
FaceHandle fh = ...;
VertexHandle v0 = ...;  // first vertex of face fh
VertexHandle v1 = ...;  // second vertex
VertexHandle v2 = ...;  // third vertex

Eigen::Vector3f p0 = position->to_eigen<3>(v0).cast<float>();
Eigen::Vector3f p1 = position->to_eigen<3>(v1).cast<float>();
Eigen::Vector3f p2 = position->to_eigen<3>(v2).cast<float>();

Eigen::Vector3f e0 = p1 - p0;
Eigen::Vector3f e1 = p2 - p0;
Eigen::Vector3f n = e0.cross(e1).normalized();
face_normal.from_eigen<3>(fh, n);
```

??? note "`to_eigen<N>(handle)` / `from_eigen<N>(handle, value)`"
    Converts between one attribute element and an Eigen vector type. `N` must match the attribute component count.

Equivalent GLM flow:

```cpp
glm::vec3 p0 = position->to_glm<3>(v0);
glm::vec3 p1 = position->to_glm<3>(v1);
glm::vec3 p2 = position->to_glm<3>(v2);

glm::vec3 n = glm::normalize(glm::cross(p1 - p0, p2 - p0));
face_normal->from_glm<3>(fh, n);
```

??? note "`to_glm<N>(handle)` / `from_glm<N>(handle, value)`"
    Converts between one attribute element and a GLM vector type. `N` must match the attribute component count.

Matrix conversion utilities that convert the attribute to/from a [dense matrix](../matrix/dense_matrices.md):

```cpp
auto mat = face_normal->to_matrix();
face_normal->from_matrix(mat);
```

??? note "`to_matrix()` / `from_matrix(mat)`"
    Converts an attribute to/from a dense matrix representation.

---

## **Metadata and Lifecycle Reference**

Use these APIs when you need to inspect shape/layout/allocation or release memory.

Correct shape semantics:

- `rows()` = number of mesh elements represented by the attribute
- `size()` = number of mesh elements (same concept as `rows()`, **not** `rows() * cols()`)
- `cols()` = number of components per element
- `get_num_attributes()` = per-element component count (same concept as `cols()`)

```cpp
auto n_elements = attr->size();
auto n_rows = attr->rows();
auto n_components = attr->cols();
auto n_attributes = attr->get_num_attributes();
```

??? note "`get_name() const`"
    Returns the attribute name used at creation time.

??? note "`rows() const`, `cols() const`, `size() const`, `get_num_attributes() const`"
    Reports shape information using the semantics described above.

Layout, footprint, and allocation checks:

```cpp
auto layout = attr->get_layout();
auto bytes = attr->get_total_bytes();
bool on_host = attr->is_host_allocated();
bool on_device = attr->is_device_allocated();
bool empty = attr->is_empty();
```

??? note "`get_layout() const`"
    Returns the memory layout (`SoA` or `AoS`) used by the attribute.

??? note "`get_total_bytes() const`"
    Returns the total allocated byte footprint for this attribute.

??? note "`get_allocated() const`, `is_host_allocated() const`, `is_device_allocated() const`, `is_empty() const`"
    Reports allocation state on host/device.

Release storage when you no longer need the data:

```cpp
attr->release(LOCATION_ALL);  // Free host and device storage
```

??? note "`release(location = LOCATION_ALL)`"
    Releases allocated storage in the requested location(s).