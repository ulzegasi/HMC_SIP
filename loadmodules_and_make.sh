#!/bin/bash
module load USS/2020
module load gcc/7.3.0
module load glib/2.56.3-py3.6-pe5.26
module load cmake/3.15.4-pe5.26
echo "modules loaded"
cmake CMakeLists.txt
# > cmake CMakeLists.txt -> to generate make file
make
# > make -> to compile
echo "make DONE"
