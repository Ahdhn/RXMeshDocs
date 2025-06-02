# Building

## Prerequisites 

The code can be compiled on Ubuntu, Windows, and WSL providing that CUDA (>=11.1.0) is installed. To run the executable(s), an NVIDIA GPU should be installed on the machine. 



## Dependencies
- [OpenMesh](https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh) to verify the applications against reference CPU implementation
- [RapidJson](https://github.com/Tencent/rapidjson) to report the results in JSON file(s)
- [GoogleTest](https://github.com/google/googletest) for unit tests
- [spdlog](https://github.com/gabime/spdlog) for logging
- [glm](https://github.com/g-truc/glm.git) for small vectors and matrices operations 
- [Eigen](https://gitlab.com/libeigen/eigen) for small vectors and matrices operations 
- [Polyscope ](https://github.com/nmwsharp/polyscope) for visualization  
- [cereal](https://github.com/USCiLab/cereal.git) for serialization 
- [METIS](https://github.com/KarypisLab/METIS) for experimentation with graph partitioning


## Compilation

All the dependencies are installed automatically! To compile the code:

```bash
git clone https://github.com/owensgroup/RXMesh.git
cd RXMesh
mkdir build 
cd build 
cmake ../
cmake --build . --config Release --parallel 8
```

The final step will compile all the applications and unit tests. Alternatively, you may choose to compile specific target as 

```
cmake --build . --config Release --parallel 8
```