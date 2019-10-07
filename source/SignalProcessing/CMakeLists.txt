cmake_minimum_required(VERSION 2.6.4)

project(SignalProcessing)

find_package(CUDA 9.0 QUIET)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../3rdParty/armadillo-code/include)
include_directories(../3rdParty/fftw/build/include)
include_directories(../Utils/include)
include_directories(include)

if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    file(GLOB_RECURSE
        source_files
        src/*
        include/*)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++14;-DUSE_CUDA;--expt-relaxed-constexpr;-Xptxas;-O3")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_62,code=sm_62;-rdc=true")
    set(CUDA_SEPARABLE_COMPILATION ON)


    cuda_add_library(SignalProcessing
        STATIC
        ${source_files})
    target_link_libraries(SignalProcessing
        armadillo
        ${openblas_LIBRARY}
        ${fftw_LIBRARY}
        gfortran
        ${CUDA_LIBRARIES}
        ${CUDA_cudadevrt_LIBRARY})

    add_definitions(-DUSE_CUDA)

else (CUDA_FOUND)

    file(GLOB_RECURSE
        source_files
        src/AnalysisDispatcher.cpp
        src/SignalProcessor.cpp
        src/SpecificSignalProcessor.cpp
        src/Analysis/*
        src/Filters/*
        src/Parameters/*
        include/SignalProcessing/AnalysisDispatcher.h
        include/SignalProcessing/ProcessingDataType.h
        include/SignalProcessing/SignalProcessor.h
        include/SignalProcessing/SpecificSignalProcessor.h
        include/SignalProcessing/Analysis/*
        include/SignalProcessing/Filters/*
        include/SignalProcessing/Parameters/*)

    add_library(SignalProcessing
        STATIC
        ${source_files})

    target_link_libraries(SignalProcessing
        armadillo
        ${openblas_LIBRARY}
        ${fftw_LIBRARY}
        gfortran)

endif (CUDA_FOUND)

set_property(TARGET SignalProcessing PROPERTY CXX_STANDARD 14)
add_dependencies(SignalProcessing fftw)

assign_source_group(${source_files})

add_subdirectory(test)
