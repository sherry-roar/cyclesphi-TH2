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
#ifndef __KERNEL_UTIL_H__
#define __KERNEL_UTIL_H__

#  ifdef _WIN32
int setenv(const char *envname, const char *envval, int overwrite);
void usleep(__int64 usec);
#else
#include <unistd.h>
#endif

void util_save_bmp(int offset,
                   int stride,
                   int tile_x,
                   int tile_y,
                   int tile_h,
                   int tile_w,
                   int pass_stride,
                   int end_samples,
                   char *buffer,
                   char *pixels,
                   int step);

void util_save_yuv(
  unsigned char* dst_y, unsigned char* dst_u, unsigned char* dst_v, int width, int height);

int util_get_count_from_env_array(const char *env_value);
int util_get_int_from_env_array(const char *env_value, int id);

void util_rgb_to_yuv_i420_stereo(
  unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

void util_rgb_to_yuv_i420(
  unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

void util_yuv_i420_to_rgba(
  unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

char* util_rgb_to_xor_rle(char* source, int tile_h, int tile_w, size_t &res_size);
char* util_xor_rle_to_rgb(char* source, int tile_h, int tile_w, size_t res_size);

unsigned int util_hash_uint2(unsigned int kx, unsigned int ky);

#endif /* __KERNEL_UTIL_H__ */
