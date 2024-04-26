add_executable(vector vector.cu)
target_compile_features(vector PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(vector PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(vector PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
target_link_libraries(vector PRIVATE ${PROJECT_NAME})

