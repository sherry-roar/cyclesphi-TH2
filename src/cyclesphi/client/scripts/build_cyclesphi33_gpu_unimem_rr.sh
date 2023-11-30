#!/bin/bash

# modules for Karolina cluster
ml Mesa/22.2.4-GCCcore-12.2.0
ml CMake/3.24.3-GCCcore-12.2.0
ml CUDA/12.0.0
ml GCC/12.2.0
ml intel/2022b

ROOT_DIR=${PWD}

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/cyclesphi33_gpu_unimem_rr
src=${ROOT_DIR}/src

export CC='gcc'
export CXX='g++'

#-----------blender_client--------------
mkdir ${ROOT_DIR}/build/cyclesphi33_gpu_unimem_rr
cd ${ROOT_DIR}/build/cyclesphi33_gpu_unimem_rr

make_d="${src}/cyclesphi/client"

######################
make_d="${make_d} -DWITH_CLIENT_CPU=OFF"
make_d="${make_d} -DWITH_CLIENT_CUDA=ON"

make_d="${make_d} -DWITH_CLIENT_UNIMEM=ON"
make_d="${make_d} -DENABLE_INC_SAMPLES=ON"
make_d="${make_d} -DENABLE_STEP_SAMPLES=ON"
make_d="${make_d} -DENABLE_LOAD_BALANCE=ON"

make_d="${make_d} -DWITH_CLIENT_FILE=ON"
make_d="${make_d} -DWITH_CLIENT_MPI_FILE=ON"
make_d="${make_d} -DCLIENT_MPI_LOAD_BALANCING_SAMPLES=ON"

make_d="${make_d} -DCUDA_BINARIES_ARCH:STRING=sm_80"
make_d="${make_d} -DCUDA_CUSTOM_LIBRARIES=$CUDA_ROOT/lib64/stubs/libcuda.so;$CUDA_ROOT/lib64/libcudart.so"

make_d="${make_d} -DWITH_CLIENT_MPI_GCC=ON"
make_d="${make_d} -DMPI_CXX_HEADER_DIR=/apps/all/impi/2021.7.1-intel-compilers-2022.2.1/mpi/2021.7.1/include"
make_d="${make_d} -DMPI_LIBRARIES=/apps/all/impi/2021.7.1-intel-compilers-2022.2.1/mpi/2021.7.1/lib/release/libmpi.so"

make_d="${make_d} -DWITH_CPU_SSE=OFF"
make_d="${make_d} -DWITH_CPU_AVX=OFF"
make_d="${make_d} -DWITH_CPU_AVX2=OFF"

make_d="${make_d} -DWITH_CLIENT_CUDA_STREAM=ON"

make_d="${make_d} -DWITH_CUDA_STAT=OFF"
make_d="${make_d} -DWITH_CUDA_STATv2=OFF"
make_d="${make_d} -DWITH_CUDA_STATv2_LB=OFF"

make_d="${make_d} -DWITH_CLIENT_FILE_MMAP=ON"
make_d="${make_d} -DWITH_CPU_IMAGE=ON"

######################

make_d="${make_d} -DCMAKE_BUILD_TYPE=Release"
make_d="${make_d} -DCMAKE_INSTALL_PREFIX=${output}"

cmake ${make_d}
make -j 64 install

### copy cubins to install dir
cp ${ROOT_DIR}/build/cyclesphi33_gpu_unimem_rr/cycles_cuda/*.cubin ${output}/lib/.

