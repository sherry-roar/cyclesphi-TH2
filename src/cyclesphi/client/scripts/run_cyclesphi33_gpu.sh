#!/bin/bash

# cd /mnt/projX/open-XX-XX/project-cyclesphi

# modules for Karolina cluster
ml Mesa/22.2.4-GCCcore-12.2.0
ml CUDA/12.0.0
ml GCC/12.2.0
ml intel/2022b

########################################

ROOT=$PWD
CACHE_DIR=$ROOT/data

export CLIENT_FILE_KERNEL_GLOBAL=${CACHE_DIR}/cache.kg
export CLIENT_FILE_CYCLES_BUFFER=${CACHE_DIR}/cache.bf
export CLIENT_FILE_CYCLES_BMP=${CACHE_DIR}/cache.bmp

export KERNEL_CUDA_CUBIN=${ROOT}/install/cyclesphi33_gpu/lib/kernel_sm_80.cubin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROOT}/install/cyclesphi33_gpu/lib

mpirun -env DEBUG_SAMPLES=1024 -env DEBUG_STEP_SAMPLES=1 $ROOT/install/cyclesphi33_gpu/bin/blender_client
