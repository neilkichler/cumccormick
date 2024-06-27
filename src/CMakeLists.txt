set(INCLUDE_PATH "${cumccormick_SOURCE_DIR}/include")
set(LIB_PATH "${INCLUDE_PATH}/cumccormick")

set(headers 
    "${LIB_PATH}/arithmetic/basic.cuh"
    "${LIB_PATH}/cumccormick.cuh"
    "${LIB_PATH}/cumccormick.h"
    "${LIB_PATH}/format.h"
    "${LIB_PATH}/mccormick.h"
)

add_library(cumccormick "main.cu" ${headers})

target_include_directories(
    cumccormick PUBLIC $<BUILD_INTERFACE:${INCLUDE_PATH}>
                      $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# C++20 is required for all users of the library
target_compile_features(cumccormick PUBLIC cxx_std_20 cuda_std_20)

source_group(
    TREE "${INCLUDE_PATH}"
    PREFIX "Header Files"
    FILES ${headers})

include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY https://github.com/neilkichler/cuinterval.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cuinterval)

target_link_libraries(cumccormick PUBLIC cuinterval)

FetchContent_Declare(
  PackageProject
  GIT_REPOSITORY https://github.com/TheLartians/PackageProject.cmake.git
  GIT_TAG v1.11.2
)
FetchContent_MakeAvailable(PackageProject)

packageProject(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    BINARY_DIR ${PROJECT_BINARY_DIR}
    INCLUDE_DIR ${INCLUDE_PATH}
    INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    DEPENDENCIES cuinterval
)
