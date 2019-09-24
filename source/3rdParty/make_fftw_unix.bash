#!/bin/bash

if [ ! -f ./build/lib/libfftw3f.a ]; then
    ./configure --enable-float --prefix=$PWD/build && make && make install
fi

if [ ! -f ./build/lib/libfftw3.a ]; then
    ./configure --prefix=$PWD/build && make && make install
fi
