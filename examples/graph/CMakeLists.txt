add_executable(capture capture.cu)
target_compile_features(capture PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(capture PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(capture PRIVATE ${PROJECT_NAME} warnings)

add_executable(manual manual.cu)
target_compile_features(manual PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(manual PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(manual PRIVATE ${PROJECT_NAME} warnings)


add_executable(kernel kernel.cu)
target_compile_features(kernel PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(kernel PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(kernel PRIVATE ${PROJECT_NAME} warnings)

add_executable(kernel_ndim kernel_ndim.cu)
target_compile_features(kernel_ndim PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(kernel_ndim PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(kernel_ndim PRIVATE ${PROJECT_NAME} warnings)

add_executable(capture_ndim capture_ndim.cu)
target_compile_features(capture_ndim PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(capture_ndim PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(capture_ndim PRIVATE ${PROJECT_NAME} warnings)
