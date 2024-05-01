add_executable(capture capture.cu)
target_compile_features(capture PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(capture PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(capture PRIVATE ${PROJECT_NAME})

add_executable(manual manual.cu)
target_compile_features(manual PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(manual PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(manual PRIVATE ${PROJECT_NAME})
