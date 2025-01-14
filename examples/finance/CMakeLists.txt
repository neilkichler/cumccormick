add_executable(blackscholes blackscholes.cu)
target_compile_features(blackscholes PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(blackscholes PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(blackscholes PRIVATE ${PROJECT_NAME} warnings)
