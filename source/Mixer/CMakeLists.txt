cmake_minimum_required(VERSION 2.6.4)

project(Mixer)

set(LIBRARY_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories(../3rdParty/fftw/build/include)
include_directories(../Utils/include)
include_directories(../SignalProcessing/include)
include_directories(../Communication/include)
include_directories(../Uniformization/include)
include_directories(include)

file(GLOB_RECURSE
    source_files
    src/*
    include/*)

add_executable(Mixer
    ${source_files})

set_property(TARGET Mixer PROPERTY CXX_STANDARD 17)

add_custom_command(TARGET Mixer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:Mixer>/resources)

add_library(Mixer_Private
    STATIC
    ${source_files})

set_property(TARGET Mixer_Private PROPERTY CXX_STANDARD 17)


if (UNIX)
    target_link_libraries(Mixer
        Utils
        SignalProcessing
        Communication
        Uniformization
        asound
        pthread
        armadillo
        ${openblas_LIBRARY}
        gfortran)

    target_link_libraries(Mixer_Private
        Utils
        SignalProcessing
        Communication
        Uniformization
        asound
        pthread
        armadillo
        ${openblas_LIBRARY}
        gfortran)
else (UNIX)
    target_link_libraries(Mixer
        Utils
        SignalProcessing
        Communication
        Uniformization
        armadillo
        ${openblas_LIBRARY}
        gfortran)

    target_link_libraries(Mixer_Private
        Utils
        SignalProcessing
        Communication
        Uniformization
        armadillo
        ${openblas_LIBRARY}
        gfortran)
endif (UNIX)

assign_source_group(${source_files})

add_subdirectory(test)
