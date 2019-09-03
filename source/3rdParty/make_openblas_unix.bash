#!/bin/bash

f [ ! -f ./libopenblas.a ]; then
    make USE_THREAD=0
fi
