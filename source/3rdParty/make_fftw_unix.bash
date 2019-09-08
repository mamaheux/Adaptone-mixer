#!/bin/bash

if [ ! -d ./build ]; then
    ./configure --enable-float --prefix=$PWD/build && make && make install
fi
