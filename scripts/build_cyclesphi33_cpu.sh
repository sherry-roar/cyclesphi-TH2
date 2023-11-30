#!/bin/bash

# modules for Karolina cluster
# ml Mesa/22.2.4-GCCcore-12.2.0
# ml CMake/3.24.3-GCCcore-12.2.0
# ml CUDA/12.0.0
# ml GCC/12.2.0
# ml intel/2022b

ROOT_DIR=${PWD}

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/cyclesphi33_cpu
src=${ROOT_DIR}/src

export CC='mpicc'
export CXX='mpicxx'

# export CC='gcc'
# export CXX='g++'

#-----------blender_client--------------
mkdir ${ROOT_DIR}/build/cyclesphi33_cpu
cd ${ROOT_DIR}/build/cyclesphi33_cpu

make_d="${src}/cyclesphi/client"

######################
make_d="${make_d} -DWITH_CLIENT_CPU=ON"

make_d="${make_d} -DWITH_CLIENT_RENDERENGINE=OFF"

make_d="${make_d} -DWITH_OPENMP=ON"
make_d="${make_d} -DWITH_CLIENT_FILE=ON" # empty comm func
make_d="${make_d} -DWITH_CLIENT_MPI_FILE=ON"
make_d="${make_d} -DCLIENT_MPI_LOAD_BALANCING_SAMPLES=ON" # changed into off by MrR
########
make_d="${make_d} -DWITH_CLIENT_MPI_FILE_1=ON"

make_d="${make_d} -DWITH_CLIENT_MPI=OFF" # empty comm func
make_d="${make_d} -DWITH_CLIENT_MPI_SOCKET=OFF" # gather/resv**
make_d="${make_d} -DWITH_SOCKET_UDP=OFF"
make_d="${make_d} -DWITH_CLIENT_SOCKET=OFF" # NET_NAMESPACE::send_data_data
#######
make_d="${make_d} -DWITH_CPU_AVX=ON"
make_d="${make_d} -DWITH_CPU_AVX2=ON"

make_d="${make_d} -DWITH_CLIENT_MPI_GCC=ON"
make_d="${make_d} -DMPI_CXX_HEADER_DIR=/usr/lib/x86_64-linux-gnu/openmpi/include"
make_d="${make_d} -DMPI_LIBRARIES=/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so"
######################

make_d="${make_d} -DCMAKE_BUILD_TYPE=debug"
make_d="${make_d} -DCMAKE_INSTALL_PREFIX=${output}"

cmake ${make_d}

make -j 8 install
