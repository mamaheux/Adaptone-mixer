cmake_minimum_required(VERSION 2.6.4)

project(Uniformization)

find_package(Boost COMPONENTS system REQUIRED)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../3rdParty/armadillo-code/include)
include_directories(../Utils/include)
include_directories(../SignalProcessing/include)
include_directories(${Boost_INCLUDE_DIR})
include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)

add_library(Uniformization
    STATIC
    ${source_files})

set_property(TARGET Uniformization PROPERTY CXX_STANDARD 17)

target_link_libraries(Uniformization
    Utils
    SignalProcessing
    armadillo
    ${openblas_LIBRARY}
    gfortran
    ${Boost_LIBRARIES})

assign_source_group(${source_files})

add_subdirectory(test)
