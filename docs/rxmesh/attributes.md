# **Attributes** 


Attributes are used to manage and manipulate data associated with mesh elements (vertices, edges, faces). To allocate attributes, see [Attributes Management](attributes_management.md). This section covers how to use attributes after allocation and what operations are available.

Each attribute is defined by:

- **Handle type**: Which mesh element the attribute is attached to (e.g., `VertexAttribute`, `EdgeAttribute`, `FaceAttribute`)

- **Data type**: The type of values stored (e.g., `float`, `int`, `bool`, or user-defined structs that are trivially copyable)

For multi-dimensional attributes (e.g., RGB color), we recommend using a scalar type with an explicit size:

```cpp
rx.add_vertex_attribute<float>("Color", 3) 
```

The attribute layout can be either Struct of Arrays (SoA) or Array of Structs (AoS), with SoA being the default. SoA typically leads to better performance on GPU. More details are available [here](https://en.wikipedia.org/wiki/AoS_and_SoA).


---
## **Per-Attribute Operations**

These operations affect the entire attribute.

- Check allocation:

```cpp
attr.get_allocated() 
```
Returns where the attribute is currently allocated (`HOST`, `DEVICE`, or both).

```cpp
attr.is_host_allocated()
attr.is_device_allocated()
```

To check if the attribute is allocated on host or device. 

- Check dimensions:

`attr.rows()` — number of mesh elements
`attr.cols()` — number of attribute components
`attr.size()` — total number of elements (`rows × cols`)

- Convert to and from matrices (host-side only):

```cpp
attr.to_matrix()
```

Converts the attribute to a host-side dense matrix (e.g., for use in Eigen or for export).

```cpp
attr.from_matrix(mat)
```

Copies data from a host-side matrix into the attribute.

- Memory movement:

``` cpp
attr.move(HOST, DEVICE)
```

Moves memory between host and device. If target is not allocated, it will be allocated first.

- Deep copy from another attribute:

```cpp
attr.copy_from(other_attr, HOST, DEVICE)
```

Performs a location-aware deep copy. See the copy semantics in the API notes.

- Reset values:

```cpp 
attr.reset(0.0f, DEVICE) 
```

Sets all attribute entries to a fixed value.


## **Element-Wise Access**

These operations allow access to or mutation of individual entries using handles or indices.

- Direct index access:

```cpp
attr(i, j) 
```
Access the j-th component of the i-th element (e.g., vertex or face). j defaults to 0.

- Handle-based access:

```cpp 
attr(handle, j)
```

Use a HandleT (e.g., VertexHandle) and access the j-th attribute.

- Interfacing with GLM and Eigen:
    - To read:
```cpp
attr.to_glm<N>(handle) 
attr.to_eigen<N>(handle)
```
    - To write:    
```cpp
attr.from_glm<N>(handle, glm_vec)
attr.from_eigen<N>(handle, eigen_vec)
```

`N` must match the number of attribute components.