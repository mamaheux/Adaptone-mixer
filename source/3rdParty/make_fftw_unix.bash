#!/bin/bash

if [ ! -d ./install ]; then
    ./configure --enable-float --enable-type-prefix && make && export DESTDIR="$PWD/install" && make install
fi
