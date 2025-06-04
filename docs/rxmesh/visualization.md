# ** Visualization **

RXMesh integrates with [Polyscope](https://polyscope.run/) to visualize triangle meshes and their associated attributes. This allows users to quickly inspect the geometry and data produced during computation. To use Polyscope with RXMesh, ensure that it is enabled at configuration time by setting the CMake option `RX_USE_POLYSCOPE=ON`. This is turned on by default:

```bash 
cd build
cmake -DRX_USE_POLYSCOPE=ON ..
```

---

## **Built-in Integration**
RXMesh provides a built-in interface for Polyscope. After constructing an `RXMeshStatic` object, the user can access a Polyscope mesh representation using:

```cpp
auto polyscope_mesh = rx.get_polyscope_mesh()
```

This object acts as the bridge between RXMesh and Polyscope’s visualization routines. In addition, RXMesh implements the necessary functionalities to pass attributes to Polyscope—thanks to Polyscope's [data adaptors](https://polyscope.run/data_adaptors). For more information about Polyscope's different visualization options, please checkout Polyscope's [Surface Mesh documentation](https://polyscope.run/structures/surface_mesh/basics/)

--- 
## **Host Requirement**
Before any attribute can be visualized, it must be available on the host. RXMesh stores attributes on either the host, the device, or both. Since Polyscope operates on CPU-side data, attributes must be explicitly moved to the host using:

```cpp
attribute.move(DEVICE, HOST)
```

Failing to do this may result in visualization discrepancies—for example, displaying outdated or incorrect results on the host that differ from what was actually computed on the device

---
## **Example**

The following example shows how to compute and visualize a per-vertex color attribute:

```cpp
RXMeshStatic rx("mesh.obj")

// Add vertex color attribute with 3 channels
auto vertex_color = rx.add_vertex_attribute<float>("vColor", 3)

// Populate vertex color on the DEVICE
// ...

// Move the data to the HOST for visualization
vertex_color.move(DEVICE, HOST)

// Retrieve Polyscope mesh instance from RXMesh
auto polyscope_mesh = rx.get_polyscope_mesh()

// Add the vertex color attribute to Polyscope
polyscope_mesh->addVertexColorQuantity("vColor", vertex_color)

// Launch the viewer
polyscope::show()
```