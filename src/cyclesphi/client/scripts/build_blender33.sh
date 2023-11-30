#!/bin/bash

# modules for Karolina cluster
ml Mesa/22.2.4-GCCcore-12.2.0
ml CMake/3.24.3-GCCcore-12.2.0
ml CUDA/12.0.0
ml GCC/12.2.0

ROOT_DIR=${PWD}

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/blender33_cyclesphi
src=${ROOT_DIR}/src

export CC='gcc'
export CXX='g++'

#-----------blender--------------
mkdir ${ROOT_DIR}/build/blender33_cyclesphi
cd ${ROOT_DIR}/build/blender33_cyclesphi

make_d="-C${src}/cyclesphi/build_files/cmake/config/blender_release.cmake ${src}/cyclesphi"

make_d="${make_d} -DWITH_CYCLES_DEVICE_ONEAPI=OFF"
make_d="${make_d} -DWITH_CYCLES_DEVICE_CLIENT=ON"
make_d="${make_d} -DWITH_CLIENT_FILE=ON"
make_d="${make_d} -DWITH_CLIENT_CUDA=ON"
make_d="${make_d} -DWITH_CLIENT_RAAS=ON"

make_d="${make_d} -DCYCLES_CUDA_BINARIES_ARCH:STRING=sm_80"
make_d="${make_d} -DWITH_CYCLES_CUDA_BINARIES=OFF"
make_d="${make_d} -DCMAKE_EXE_LINKER_FLAGS=-fopenmp"

make_d="${make_d} -DCMAKE_INSTALL_PREFIX=${output}"
make_d="${make_d} -DCMAKE_BUILD_TYPE=Release"

cmake ${make_d}

make -j 64 install
