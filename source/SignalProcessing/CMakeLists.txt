cmake_minimum_required(VERSION 2.6.4)

project(SignalProcessing)

find_package(CUDA 9.0)

if (CUDA_FOUND)

    set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(../Utils/include)
    include_directories(include)

    link_directories(${CUDA_LIBRARY_DIRS})

    file(GLOB_RECURSE
        source_files
        src/*
        include/*)

    cuda_add_library(SignalProcessing
        STATIC
        ${source_files})

    set_property(TARGET SignalProcessing PROPERTY CXX_STANDARD 14)

    target_link_libraries(SignalProcessing
        armadillo)

    assign_source_group(${source_files})

    add_subdirectory(test)

endif (CUDA_FOUND)
