#!/bin/bash

# modules for Karolina cluster
# ml Mesa/22.2.4-GCCcore-12.2.0
# ml CUDA/12.0.0
# ml GCC/12.2.0

ROOT=$PWD
CACHE_DIR=$ROOT/data

mkdir -p $CACHE_DIR

export CLIENT_FILE_ACTION=PRE
export CLIENT_FILE_KERNEL_GLOBAL=${CACHE_DIR}/cache.kg

# $ROOT/install/blender33_cyclesphi/blender -b $ROOT/src/lib/benchmarks/cycles/classroom/classroom.blend --python $ROOT/src/cyclesphi/client/data/over.py -y -x 1 -o $CACHE_DIR/#### -F PNG -f 1
$ROOT/install/blender33_cyclesphi/blender -b $ROOT//src/lib/benchmarks/cycles/bmw27/bmw27.blend --python $ROOT/src/cyclesphi/client/data/over.py -y -x 1 -o $CACHE_DIR/#### -F PNG -f 1
