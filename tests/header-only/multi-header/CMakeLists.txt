cmake_minimum_required(VERSION 3.25.2)

project(cumccormick-multi-header-test LANGUAGES CXX CUDA)

# cumccormick requires cuinterval
include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY git@github.com:neilkichler/cuinterval.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cuinterval)

add_executable(multi-header multi_header.cu)
target_include_directories(multi-header PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(multi-header PRIVATE cuinterval)
target_compile_features(multi-header PRIVATE cuda_std_20)
target_compile_options(multi-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(multi-header PROPERTIES CUDA_ARCHITECTURES native)
