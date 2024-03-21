include(FetchContent)

# Testing library
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.0.1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ut)

# Tests need to be added as executables first
add_executable(tests 
    tests.cpp
    tests_basic.cpp
    tests_basic.cu
    tests_basic.h
)

# Use C++20 in the tests
target_compile_features(tests PRIVATE cxx_std_20)
target_compile_features(tests PRIVATE cuda_std_20)

# Specify the CUDA architecture to compile to
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES native)

# Enable separate compilation
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link with main library and testing library
target_link_libraries(tests PRIVATE cumccormick ut)

# Find and link with CUDA headers
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tests PRIVATE CUDA::cuda_driver)
target_link_libraries(tests PRIVATE CUDA::cudart)

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
endif()

# Register tests for use in ctest
add_test(NAME testcuinterval COMMAND tests)

