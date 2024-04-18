add_executable(examples cuda_graphs.cu)
target_compile_features(examples PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(examples PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(examples PRIVATE ${PROJECT_NAME})
