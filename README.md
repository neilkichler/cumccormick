<h1 align='center'>CuMcCormick</h1>

CuMcCormick is a CUDA library for McCormick relaxations.

## Supported Operations

## Installation
> Please make sure that you have installed everything mentioned in the section [Build Requirements](#build-requirements).
```bash
git clone https://github.com/neilkichler/cumccormick.git
cd cumccormick
cmake --preset release
cmake --build build
cmake --install build
```

## Example
Have a look at the [examples folder](https://github.com/neilkichler/cumccormick/tree/main/examples).

## Documentation
The documentation is available [here](https://neilkichler.github.io/cumccormick).

## Build

### Build Requirements
We require C++20, CMake v3.21+, Ninja, and recent C++ and CUDA compilers.

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
