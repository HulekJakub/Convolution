cmake_minimum_required(VERSION 3.14)
project(my-convolution)

include(CheckCXXCompilerFlag)
include(ExternalProject)
include(FetchContent)

enable_testing()

find_package(Git)

if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
endif()


FetchContent_Declare(
dnnl
GIT_REPOSITORY https://github.com/intel/mkl-dnn
GIT_TAG v2.1
GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dnnl)

add_library(conv_lib "")
add_subdirectory(source/data)
add_subdirectory(source/convolution)
add_subdirectory(source/convolution_quant)
add_subdirectory(source/utils)

add_executable(my-convolution ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp) 

target_link_libraries(conv_lib PUBLIC dnnl)
target_link_libraries(my-convolution conv_lib)

