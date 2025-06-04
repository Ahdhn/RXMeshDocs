# **RXMeshStatic**

`RXMeshStatic` is the main class for representing and processing static triangle meshes with fixed connectivity—meshes that do not change topology at runtime. This class constructs a compact, patch-based internal representation of the input mesh and exposes interfaces for managing mesh attributes, launching element-wise operations, and querying connectivity.

The input to `RXMeshStatic` is typically a `.obj` file containing a triangle mesh. During construction, RXMesh parses the mesh and builds a data structure optimized for GPU execution. RXMesh uses [handles](handles.md) to represent mesh elements, which can be converted to *continuous* indices when needed. For details, see [Indexing](#indexing).

Users interact with `RXMeshStatic` by defining attributes associated with vertices, edges, or faces. Each attribute is defined by its element type, value type, and dimensionality. Attributes can be allocated on either the host, device, or both and transferred explicitly between them. See [Attribute Management](#attribute-management) for more.

Mesh operations are expressed either through parallel iteration over individual elements (e.g., `for_each_vertex`) or via query kernels (e.g., `run_query_kernel<Op::FV>`) that provide access to local neighborhoods. These kernels execute entirely on the GPU and support computations that depend on adjacent elements. See [Operations](#operations) for more information.


<!-- <hr style="border: none; border-top: 1px solid #888;"/>

<hr style="border: none; border-top: 2px solid #ccc;" />


<hr style="border: none; border-top: 1px solid #ccc; margin: 2em 0;" />

---  -->

<hr style="border: none; border-top: 1px solid #888;"/>

## **1. Initialization**

`RXMeshStatic` can be constructed in two ways, depending on whether the input mesh is stored in a file or directly provided as face connectivity in memory.

---

### **1.1 From an `.obj` File**

```cpp
RXMeshStatic(const std::string obj_file_path,
             const std::string patcher_file = "",
             const uint32_t    patch_size   = 512);
```

This constructor takes a path to a triangle mesh stored as a `.obj` file and builds a patch-based internal representation optimized for GPU execution.

- `obj_file_path`: Path to the input .obj mesh file.

- `patcher_file`: Optional. If provided, RXMesh will use the patch layout stored in this file instead of generating new patches.

- `patch_size`: Controls the target average number of faces per patch. The default is 512.

#### Patch-Based Construction and Determinism
RXMesh uses a patch-based decomposition of the mesh to improve data locality and enable efficient parallel processing. By default, this patching process is run at every construction and includes randomness (e.g., random starting points or ordering) which can lead to non-deterministic patch layouts. This does not affect correctness but can make debugging more difficult.

To make the patching deterministic, users can supply a precomputed patch layout via the `patcher_file` argument. This file stores the result of a previous patching run and allows RXMesh to reuse the same layout. To generate such a file, call:

```c++
RXMeshStatic rx("mesh.obj");
rx.save("patches.rx");
```
You can then pass `patches.rx` as the `patcher_file` argument in subsequent constructions.

--- 

### **1.2 From memory**
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

<hr style="border: none; border-top: 1px solid #888;"/>

## **2. Attributes Management**

This section explains how to define, check, and remove attributes in `RXMeshStatic`. Attributes in RXMesh are strongly typed and always associated with a specific mesh element (vertex, edge, or face). You can create attributes from scratch, load them from data in memory, or copy the shape and layout from an existing attribute.

> For manipulating attribute values—access, memory movement, or math—see the [Attributes](attributes.md) section.

---

### **2.1. Adding Attributes**

#### From Scratch

Used when you want to allocate a new attribute with no initial values. You specify the number of values per element, where they should be allocated (host/device), and the layout (SoA or AoS).

```cpp
rx.add_vertex_attribute<float>("vColor", 3);   // RGB color
rx.add_edge_attribute<int>("eFlags", 1);       // One integer flag per edge
rx.add_face_attribute<double>("area", 1);      // Scalar per face
```

Relevant functions:

- `add_vertex_attribute<T>(name, num_attributes, location, layout)`
- `add_edge_attribute<T>(...)`
- `add_face_attribute<T>(...)`
- `add_attribute<T, HandleT>(...)` — generic version that uses the handle type.

---

#### From Existing Data

Used when you already have attribute values stored in memory (e.g., parsed from a file or generated by preprocessing).

```c++
// Per-vertex 3D coordinates
std::vector<std::vector<float>> coords = ...;
rx.add_vertex_attribute(coords, "vertex_positions");

// Per-face scalar value
std::vector<int> face_labels = ...;
rx.add_face_attribute(face_labels, "fLabels");
```

Functions:

- `add_vertex_attribute<T>(std::vector<std::vector<T>>, name, layout)`
- `add_face_attribute<T>(std::vector<std::vector<T>>, name, layout)`
- `add_vertex_attribute<T>(std::vector<T>, name, layout)`
- `add_face_attribute<T>(std::vector<T>, name, layout)`

These methods copy the input data to both host and device and allocate storage accordingly.

---

#### From Existing Attribute (Same Shape)

Used when you want to allocate an attribute that has the same layout, memory location, and shape as an existing one.

``` c++
auto new_color = rx.add_vertex_attribute_like<float>("vColor2", old_color);
```

Functions:

- `add_vertex_attribute_like<T>(name, other)`
- `add_edge_attribute_like<T>(...)`
- `add_face_attribute_like<T>(...)`
- `add_attribute_like<T, HandleT>(...)` — type inferred from `HandleT`

---

### **2.2. Checking for Existence**

To check if an attribute with a given name is already defined:

``` c++
if (rx.does_attribute_exist("vColor")) {
    // Safe to reuse or modify
}
```

Function:

- `does_attribute_exist(name)`

---

### **2.3. Removing Attributes**

To delete an attribute from RXMesh and release its memory:

``` c++
rx.remove_attribute("vColor");
```

Function:

- `remove_attribute(name)`

<hr style="border: none; border-top: 1px solid #888;"/>

## **3. Computations **
---

### **3.1 `for_each`**

---

### **3.2 `run_kernel` **

---

### **3.3 `run_query_kernel` **

<hr style="border: none; border-top: 1px solid #888;"/>

## **4. Indexing **

---
### **4.1 Linear ID **

---
### **4.2 `map_to_global` **

<hr style="border: none; border-top: 1px solid #888;"/>

## Boundary Vertices 

## Scale

## Bounding box

## Exporting 

## Vertex/face list