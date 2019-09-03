#!/bin/bash

f [ ! -d ./install ]; then
    ./configure --enable-float --enable-type-prefix && make && export DESTDIR="${CMAKE_CURRENT_SOURCE_DIR}/fftw/install" && make install
fi
