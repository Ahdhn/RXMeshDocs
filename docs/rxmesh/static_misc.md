# **Misc**

`RXMeshStatic` also offers a set of utility functions for common mesh-related operations:

## ** Boundary Vertices **
To determine which vertices lie on the boundary of the mesh, first allocate a vertex attribute (e.g., `int`, `bool`, or `float`), then pass it to:

```cpp 
rx.get_boundary_vertices(boundary_attr) 
```

The function sets the first component of each entry to 1 if the corresponding vertex is on the boundary, and 0 otherwise. If `boundary_attr` is allocated on the host, the result will be moved back automatically.

## ** Scale **
To scale the mesh so that it fits within a given axis-aligned bounding box, use:

```cppp
 rx.scale(lower_bound, upper_bound)
 ```

Both `lower_bound` and `upper_bound` are of type `glm::fvec3`. This modifies the coordinates returned by `get_input_vertex_coordinates()` and can be useful for normalization or fitting meshes into canonical spaces.


## ** Bounding box **
To compute the current bounding box of the mesh:

```cpp
rx.bounding_box(lower, upper) 
```

This writes the minimum and maximum corners of the axis-aligned bounding box to `lower` and `upper` (both of type `glm::vec3`), based on the current vertex coordinates.

## ** Exporting **

RXMesh supports exporting mesh data in common formats:

### OBJ Export: 

Export the mesh as a standard `.obj` file with:

```cpp
rx.export_obj("mesh.obj", coords_attr) 
```

Here, `coords_attr` is a vertex attribute representing coordinates (typically `get_input_vertex_coordinates()` or a modified version of it).

### VTK Export

Export the mesh and its attributes to a `.vtk` file for visualization in ParaView:

```cpp
rx.export_vtk("mesh.vtk", coords_attr, attr1, attr2, ...)
```

You can pass any number of additional vertex or face attributes. Note: edge attributes are not supported.

## ** Vertex/face list **

To extract the mesh geometry as raw lists for use outside RXMesh:

### Vertex List

Convert a vertex coordinate attribute to a `std::vector<glm::vec3>`:

```cpp
rx.create_vertex_list(vertex_list, coords_attr)
```

### Face List

Convert mesh connectivity to a face list of triangle indices:

```cpp 
rx.create_face_list(face_list)
```

These lists can be useful for exporting to other libraries or performing custom processing on CPU.