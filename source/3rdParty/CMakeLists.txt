cmake_minimum_required(VERSION 2.6.4)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

if (WIN32)
    #TODO
elseif (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(openblas_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS/libopenblas.a")

    add_custom_target(openblas
        COMMAND make USE_THREAD=0
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS)
endif (WIN32)

add_subdirectory(googletest)

add_subdirectory(armadillo-code)
add_dependencies(armadillo openblas)

add_subdirectory(Websocket)
