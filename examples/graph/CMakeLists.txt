add_executable(stream_capture stream_capture.cu)
target_compile_features(stream_capture PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(stream_capture PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(stream_capture PRIVATE ${PROJECT_NAME})

