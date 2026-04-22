# **Overview**

`RXMeshStatic` is the main class for representing and processing static triangle meshes, i.e., meshes with fixed connectivity that does not change at runtime. 

During construction, `RXMeshStatic` parses the input mesh (from an `.obj` file or in-memory face list) and builds a compact, [patch-based](../getting-started/concepts.md#patches) GPU data structure optimized for parallel execution. Once constructed, it exposes the full API for mesh processing. The following sections cover:

- **[Initialization](initialization.md)**: Constructors, patching options, and how to provide vertex coordinates.
- **Attributes**: Define typed per-element data. See [Managing Attributes](managing_attributes.md) for allocation and [Working with Attributes](working_with_attributes.md) for access and manipulation.
- **Operations**: Run parallel computations over mesh elements:
    - [`for_each`](for_each.md): Apply a lambda per vertex, edge, or face that may require local neighborhoods access (e.g., face vertices, vertex one-ring).
    - [Custom Kernels](run_kernel.md): Full control with multiple queries, shared memory, and custom logic.
- **[Reductions](reduce_handle.md)**: Compute global aggregates (dot products, norms, argmin/argmax) over attributes.
- **[Visualization](visualization.md)**: Render meshes and attributes with Polyscope.
- **[Indexing](indexing.md)**: Convert between handles, linear IDs, and original input indices.
- **[Utilities](static_misc.md)**: Boundary detection, bounding box, mesh export (OBJ/VTK), and more.

For operations that change mesh topology (edge flips, splits, collapses), see [Dynamic Mesh Processing](dynamic.md).
