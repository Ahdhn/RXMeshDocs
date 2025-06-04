# ** Introduction **

`RXMeshStatic` is the main class for representing and processing static triangle meshes with fixed connectivity—meshes that do not change topology at runtime. This class constructs a compact, patch-based internal representation of the input mesh and exposes interfaces for managing mesh attributes, launching element-wise operations, and querying connectivity.

The input to `RXMeshStatic` is typically a `.obj` file containing a triangle mesh. During construction, RXMesh parses the mesh and builds a data structure optimized for GPU execution. RXMesh uses [handles](handles.md) to represent mesh elements, which can be converted to *continuous* indices when needed. For details, see [Indexing](indexing.md).

Users interact with `RXMeshStatic` by defining attributes associated with vertices, edges, or faces. Each attribute is defined by its element type, value type, and dimensionality. Attributes can be allocated on either the host, device, or both and transferred explicitly between them. See [Attribute Management](attributes_management.md) for more.

Mesh operations are expressed either through parallel iteration over individual elements (e.g., `for_each_vertex`) or via query kernels (e.g., `run_query_kernel<Op::FV>`) that provide access to local neighborhoods. These kernels execute entirely on the GPU and support computations that depend on adjacent elements. See [Operations](operations.md) for more information.


<!-- <hr style="border: none; border-top: 1px solid #888;"/>

<hr style="border: none; border-top: 2px solid #ccc;" />


<hr style="border: none; border-top: 1px solid #ccc; margin: 2em 0;" />

---  -->


<hr style="border: none; border-top: 1px solid #888;"/>

## Boundary Vertices 

## Scale

## Bounding box

## Exporting 

## Vertex/face list