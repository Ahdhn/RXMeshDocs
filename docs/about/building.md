# Building

## Prerequisites

RXMesh supports Ubuntu, Windows, and WSL. To build the library, you will need:

- **CUDA Toolkit ≥ 11.1.0**  
- A **C++17-capable compiler** (e.g., GCC 9+, MSVC 2019+, Clang 10+)
- **CMake ≥ 3.15**
- **Git**
- An **NVIDIA GPU** for running the applications

If you are using WSL, ensure your distribution is correctly configured to access the GPU via [WSL 2 and CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

On Ubuntu, to enable visualization (via Polyscope), you may need to install OpenGL-related packages:
```bash 
sudo apt-get update
sudo apt-get install -y xorg-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev
```
This ensures that GUI-based rendering works correctly on most Ubuntu systems.

## Dependencies

RXMesh relies on the following libraries:

- [OpenMesh](https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh) – reference CPU implementation
- [RapidJSON](https://github.com/Tencent/rapidjson) – output benchmarking results
- [GoogleTest](https://github.com/google/googletest) – unit testing
- [spdlog](https://github.com/gabime/spdlog) – logging and diagnostics
- [glm](https://github.com/g-truc/glm.git) – small vector/matrix operations
- [Eigen](https://gitlab.com/libeigen/eigen) – linear algebra backend
- [Polyscope](https://github.com/nmwsharp/polyscope) – mesh visualization
- [cereal](https://github.com/USCiLab/cereal.git) – serialization
- [METIS](https://github.com/KarypisLab/METIS) – mesh partitioning experiments

> All dependencies are automatically downloaded and built via CMake (no manual installation needed).

## Compilation

To compile RXMesh and its test/apps:

```bash
git clone https://github.com/owensgroup/RXMesh.git
cd RXMesh
mkdir build
cd build
cmake ..
cmake --build . --config Release --parallel 8
```

This will build all targets, including unit tests and example applications.

To build only a specific target:

```bash
cmake --build . --target <target_name> --config Release --parallel 8
```

> Replace `<target_name>` with the name of the app/test you want.



## Optional CMake Flags

You can customize the build using the following CMake options:

| Option              | Default | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `RX_USE_POLYSCOPE`  | `ON`    | Enable Polyscope for visualization.             |
| `RX_BUILD_TESTS`    | `ON`    | Build RXMesh unit tests.                        |
| `RX_BUILD_APPS`     | `ON`    | Build RXMesh example applications.              |

To disable any of these options, simply pass them to `cmake` when configuring:

```bash
cmake .. -DRX_USE_POLYSCOPE=OFF -DRX_BUILD_TESTS=OFF
```