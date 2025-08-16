cmake_minimum_required(VERSION 3.18)
project(SiftGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB SRC_CPP CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
file(GLOB SRC_CU  CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB SRC_HDR CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

add_library(SiftGPU ${SRC_CPP} ${SRC_CU} ${SRC_HDR})

target_include_directories(SiftGPU
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(SiftGPU
    PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart)

set_target_properties(SiftGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY      Shared
    POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_definitions(SiftGPU PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
