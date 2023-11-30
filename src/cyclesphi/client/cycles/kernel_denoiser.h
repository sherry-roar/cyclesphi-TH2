/*
 * Copyright 2011-2013 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Contributors: Milan Jaros, IT4Innovations, VSB - Technical University of Ostrava
 *
 */
#ifndef __KERNEL_DENOISER_H__
#define __KERNEL_DENOISER_H__

#  include <cuda.h>
#  include <cuda_runtime_api.h>

#include "kernel_cuda_device.h"
#include "kernel_queue.h"

namespace cyclesphi {
namespace kernel {

char *device_optix_create();
void device_optix_destroy(char *den_ptr);
//bool denoise_optix(char *den_ptr, cudaStream_t stream, char *input_rgb_device_pointer, int rtile_w, int rtile_h);
void device_optix_denoise_buffer(char *den_ptr,
                                 CUDAContextScope &scope,
                                 cudaStream_t cuda_stream,
                                 CUDADeviceQueue *queue,
                                 char *wtile,
                                 int pass_stride,
                                 DEVICE_PTR d_input_rgb);
}}

#endif /* __KERNEL_DENOISER_H__ */
