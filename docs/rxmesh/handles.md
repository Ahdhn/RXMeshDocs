**Handles** are the unique identifiers for vertices, edges, and faces. They are usually internally populated by RXMesh (by concatenating the patch ID and mesh element index within the patch). Handles can be used to access attributes, `for_each` operations, and query operations.

Example: Setting vertex attribute using vertex handle

```c++
auto vertex_color = ...    
VertexHandle vh; 
//...

vertex_color(vh, 0) = 0.9;
vertex_color(vh, 1) = 0.5;
vertex_color(vh, 2) = 0.6;
```