# The parent enables CUDA only when the optional SM75 backend is requested.
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "sage_sm75_kernels requires the parent to enable CUDA")
endif()
find_package(CUDAToolkit REQUIRED)
add_library(sage_sm75_kernels STATIC src/rms_norm.cu src/q4k_slots.cu)
target_include_directories(sage_sm75_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sage_sm75_kernels PUBLIC CUDA::cudart)
target_compile_features(sage_sm75_kernels PUBLIC cxx_std_17)
set_target_properties(sage_sm75_kernels PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 75
  CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

add_executable(sage_rmsnorm_parity tests/rms_norm_parity.cpp)
target_link_libraries(sage_rmsnorm_parity PRIVATE sage_sm75_kernels CUDA::cudart)
target_compile_features(sage_rmsnorm_parity PRIVATE cxx_std_17)
if(BUILD_TESTING)
  add_test(NAME sage_rmsnorm_parity COMMAND sage_rmsnorm_parity)
endif()
add_executable(sage_q4k_slots_parity tests/q4k_slots_parity.cpp)
target_link_libraries(sage_q4k_slots_parity PRIVATE sage_sm75_kernels CUDA::cudart)
if(BUILD_TESTING)
  add_test(NAME sage_q4k_slots_parity COMMAND sage_q4k_slots_parity)
endif()
