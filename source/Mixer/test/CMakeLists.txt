cmake_minimum_required(VERSION 2.6.4)

project(MixerTests)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../../3rdParty/googletest/googletest/include)
include_directories(../../3rdParty/googletest/googlemock/include)
include_directories(../../Utils/include)
include_directories(../include)
include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)
     
add_executable(MixerTests
    ${source_files})

set_property(TARGET MixerTests PROPERTY CXX_STANDARD 14)
    
target_link_libraries(MixerTests
    Utils
    Mixer_Private
    gtest
    gmock)
    
assign_source_group(${source_files})
