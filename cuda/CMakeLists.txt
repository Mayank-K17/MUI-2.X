cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(muiCUDA LANGUAGES CXX)

set(BUILD_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

include(CheckLanguage)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)
find_library(LIBRT rt)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "-O3 -std=c++11 -DMUI_CUDA")
  set_source_files_properties(mui_cuda.cpp PROPERTIES LANGUAGE CUDA)
endif()

include_directories(${MPI_INCLUDE_PATH})

add_library(muiCUDA STATIC mui_cuda.cpp)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(muiCUDA PUBLIC "-pthread")
endif()

if(CMAKE_THREAD_LIBS_INIT AND LIBRT)
  target_link_libraries(muiCUDA ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${LIBRT} ${CMAKE_THREAD_LIBS_INIT})
elseif(CMAKE_THREAD_LIBS_INIT AND NOT LIBRT)
  target_link_libraries(muiCUDA ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
elseif(LIBRT AND NOT CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(muiCUDA ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${LIBRT})
else()
  target_link_libraries(muiCUDA ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES})
endif()
