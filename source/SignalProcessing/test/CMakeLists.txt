cmake_minimum_required(VERSION 2.6.4)

project(SignalProcessingTests)

find_package(CUDA 9.0 QUIET)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../../3rdParty/googletest/googletest/include)
include_directories(../../3rdParty/googletest/googlemock/include)
include_directories(../../3rdParty/armadillo-code/include)
include_directories(../../3rdParty/fftw/build/include)
include_directories(../../Utils/include)
include_directories(../include)
include_directories(include)

if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    file(GLOB_RECURSE
        source_files
        src/*
        include/*)

    cuda_add_executable(SignalProcessingTests
        ${source_files})

    target_link_libraries(SignalProcessingTests
        SignalProcessing
        Utils
        armadillo
        ${openblas_LIBRARY}
        ${fftw_LIBRARY}
        gfortran
        gtest
        gmock
        ${CUDA_LIBRARIES})

    add_definitions(-DUSE_CUDA)

else (CUDA_FOUND)

    file(GLOB_RECURSE
        source_files
        src/main.cpp
        src/Analysis/*
        src/Filters/Design/*
        src/Parameters/*)

    add_executable(SignalProcessingTests
        ${source_files})

    target_link_libraries(SignalProcessingTests
        SignalProcessing
        Utils
        armadillo
        ${openblas_LIBRARY}
        ${fftw_LIBRARY}
        gtest
        gmock)
endif (CUDA_FOUND)

set_property(TARGET SignalProcessingTests PROPERTY CXX_STANDARD 14)

assign_source_group(${source_files})

add_custom_command(TARGET SignalProcessingTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:SignalProcessingTests>/resources)
