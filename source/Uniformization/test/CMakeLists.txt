cmake_minimum_required(VERSION 2.6.4)

project(UniformizationTests)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../../3rdParty/googletest/googletest/include)
include_directories(../../3rdParty/googletest/googlemock/include)
include_directories(../../Utils/include)
include_directories(../SignalProcessing/include)
include_directories(../include)
include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)

add_executable(UniformizationTests
    ${source_files})

set_property(TARGET UniformizationTests PROPERTY CXX_STANDARD 17)

target_link_libraries(UniformizationTests
    Uniformization
    Utils
    SignalProcessing
    armadillo
    ${openblas_LIBRARY}
    gtest
    gmock)

assign_source_group(${source_files})
