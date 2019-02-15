cmake_minimum_required(VERSION 2.6.4)

project(SignalProcessing)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)

add_library(SignalProcessing
    STATIC
    ${source_files})

set_property(TARGET SignalProcessing PROPERTY CXX_STANDARD 14)

target_link_libraries(SignalProcessing
    armadillo)

assign_source_group(${source_files})

add_subdirectory(test)
