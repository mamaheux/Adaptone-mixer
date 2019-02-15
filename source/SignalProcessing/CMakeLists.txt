cmake_minimum_required(VERSION 2.6.4)

project(SignalProcessing)

find_package(CUDA 9.0 QUIET)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../Utils/include)
include_directories(include)

if (CUDA_FOUND)

    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    file(GLOB_RECURSE
        source_files
        src/*
        include/*)

    cuda_add_library(SignalProcessing
        STATIC
        ${source_files})

    target_link_libraries(SignalProcessing
        armadillo
        ${CUDA_LIBRARIES})

    add_compile_definitions(USE_CUDA)

    add_subdirectory(test)

else (CUDA_FOUND)

    add_library(SignalProcessing
        STATIC
        include/SignalProcessing/SignalProcessor.h
        src/SignalProcessor.cpp)

    target_link_libraries(SignalProcessing
        armadillo)

endif (CUDA_FOUND)


set_property(TARGET SignalProcessing PROPERTY CXX_STANDARD 14)
assign_source_group(${source_files})
