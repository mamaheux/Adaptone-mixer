cmake_minimum_required(VERSION 2.6.4)

project(Communication)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../3rdParty/Websocket)
include_directories(../3rdParty/json/include)
include_directories(../Utils/include)
include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)

add_library(Communication
    STATIC
    ${source_files})

set_property(TARGET Communication PROPERTY CXX_STANDARD 14)

target_link_libraries(Communication
    Utils
    simple-websocket-server
    nlohmann_json)

assign_source_group(${source_files})

add_subdirectory(test)
