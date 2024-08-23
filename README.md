<h1 align='center'>CuMcCormick</h1>

CuMcCormick is a CUDA library for McCormick relaxations.

## Supported Operations
`neg`
`add`
`sub`
`mul`
`div`
`sqr`
`sqrt`
`abs`
`fabs`
`exp`
`log`
`pown`
`pow`
`cos`

`asin`
`acos`
`atan`
`sinh`
`cosh`
`tanh`
`asinh`
`acosh` 
`atanh`

`max`
`min`
`mid`
`width`
`hull`

`+`
`-`
`*`
`/`
`==`
`!=`

`inf`
`sup`


## Installation
> Please make sure that you have installed everything mentioned in the section [Build Requirements](#build-requirements).

### System-wide
```bash
git clone https://github.com/neilkichler/cumccormick.git
cd cuinterval
cmake --preset release
cmake --build build
cmake --install build
```

### CMake Project


#### [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)
```cmake
CPMAddPackage("gh:neilkichler/cumccormick@0.0.1")
```

#### [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
```cmake
include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY git@github.com:neilkichler/cumccormick.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cumccormick)
```

In either case, you can link to the library using:
```cmake
target_link_libraries(${PROJECT_NAME} PUBLIC cumccormick)
```

## Example
Have a look at the [examples folder](https://github.com/neilkichler/cumccormick/tree/main/examples).

## Documentation
The documentation is available [here](https://neilkichler.github.io/cumccormick).

## Build

### Build Requirements
We require C++20, CMake v3.25.2+, Ninja, and recent C++ and CUDA compilers.

#### Ubuntu
```bash
apt install cmake gcc ninja-build
```
#### Cluster
```bash
module load CMake CUDA GCC Ninja
```

### Build and run tests
#### Using Workflows
```bash
cmake --workflow --preset dev
```
#### Using Presets
```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```
#### Using regular CMake
```bash
cmake -S . -B build -GNinja
cmake --build build
./build/tests/tests
```
