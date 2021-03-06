cmake_minimum_required(VERSION 2.6.4)

project(UtilsTests)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../../3rdParty/googletest/googletest/include)
include_directories(../../3rdParty/googletest/googlemock/include)
include_directories(../include)
include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)

add_executable(UtilsTests
    ${source_files})

set_property(TARGET UtilsTests PROPERTY CXX_STANDARD 14)

target_link_libraries(UtilsTests
    Utils
    gtest
    gmock)

assign_source_group(${source_files})

add_custom_command(TARGET UtilsTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:UtilsTests>/resources)
