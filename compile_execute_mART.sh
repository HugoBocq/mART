#!/bin/bash

ulimit -s 62767
mpiCC -O2 -DNDEBUG --std=c++11 -g -o mART.exe mART.cpp

lamboot -v
mpiexec -n 4 ./mART.exe
lamhalt -v
