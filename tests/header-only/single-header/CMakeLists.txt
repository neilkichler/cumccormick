cmake_minimum_required(VERSION 3.25.2)

project(cumccormick-single-header-test LANGUAGES CXX CUDA) # TODO: Why does this not work when leaving out CXX?

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# cumccormick requires cuinterval
include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY git@github.com:neilkichler/cuinterval.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cuinterval)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cumccormick.cuh
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/single-header.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/single-header.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Python script to generate single-header library"
)

add_custom_target(single-header DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cumccormick.cuh)

add_executable(project-header project_header.cu)
add_dependencies(project-header single-header)
target_link_libraries(project-header PRIVATE cuinterval)
target_compile_features(project-header PRIVATE cuda_std_20)
target_compile_options(project-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(project-header PROPERTIES CUDA_ARCHITECTURES native)

add_library(cumccormick-single-header INTERFACE IMPORTED)
target_include_directories(cumccormick-single-header INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(system-header system_header.cu)
add_dependencies(system-header single-header)
target_link_libraries(system-header PRIVATE cuinterval cumccormick-single-header)
target_compile_features(system-header PRIVATE cuda_std_20)
target_compile_options(system-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(system-header PROPERTIES CUDA_ARCHITECTURES native)
