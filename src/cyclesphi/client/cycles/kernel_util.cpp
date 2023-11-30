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

#include "kernel_util.h"

#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
#endif

#include <stdlib.h>

#include <cmath>
#include <map>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include <algorithm>
#include <sstream>

#ifndef _WIN32
#  include <unistd.h>
#endif

#ifdef _WIN32
//#  include <windows.h>

int setenv(const char *name, const char *value, int overwrite)
{
#  if 1
  int errcode = 0;
  if (!overwrite) {
    size_t envsize = 0;
    errcode = getenv_s(&envsize, NULL, 0, name);
    if (errcode || envsize)
      return errcode;
  }
  return _putenv_s(name, value);
#  else
  return 1;
#  endif
}

void usleep(__int64 usec)
{
  HANDLE timer;
  LARGE_INTEGER ft;

  ft.QuadPart = -(
      10 * usec);  // Convert to 100 nanosecond interval, negative value indicates relative time

  timer = CreateWaitableTimer(NULL, TRUE, NULL);
  SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
  WaitForSingleObject(timer, INFINITE);
  CloseHandle(timer);
}

#endif

////////////////////////////////////////////////////
float util_linear2srgb(float c)
{
  if (c < 0.0031308f)
    return (c < 0.0f) ? 0.0f : c * 12.92f;
  else
    return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

float util_half_to_float(unsigned short h)
{
  float f;

  *((int *)&f) = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);

  return f;
}

unsigned char util_half_to_byte(unsigned short h)
{
  return (unsigned char)(util_linear2srgb(util_half_to_float(h)) * 255.0f);
}

float util_saturate(float a)
{
  return std::min(std::max(a, 0.0f), 1.0f);
}

void util_film_map(float irradiance[4], float scale, float result[4])
{

  float exposure = 1.0f;

  /* conversion to srgb */
  result[0] = util_linear2srgb(irradiance[0] * exposure * scale);
  result[1] = util_linear2srgb(irradiance[1] * exposure * scale);
  result[2] = util_linear2srgb(irradiance[2] * exposure * scale);

  /* clamp since alpha might be > 1.0 due to russian roulette */
  result[3] = util_saturate(irradiance[3]);
}

void util_film_float_to_byte(float color[4], unsigned char result[4])
{
  /* simple float to byte conversion */
  result[0] = (unsigned char)(util_saturate(color[0]) * 255.0f);
  result[1] = (unsigned char)(util_saturate(color[1]) * 255.0f);
  result[2] = (unsigned char)(util_saturate(color[2]) * 255.0f);
  result[3] = (unsigned char)(util_saturate(color[3]) * 255.0f);
}

void util_film_convert_to_byte(float *buffer,
                               float sample_scale,
                               int x,
                               int y,
                               int offset,
                               int stride,
                               int pass_stride,
                               unsigned char byte_result[4])
{

  /* buffer offset */
  buffer += (offset + x + y * stride) * pass_stride;

  /* map colors */
  float float_result[4];
  util_film_map(buffer, sample_scale, float_result);
  util_film_float_to_byte(float_result, byte_result);
}

// written by Evercat

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
                   int step)
{
  const char *fn = std::getenv("CLIENT_FILE_CYCLES_BMP");
  if (fn == NULL) {
    printf("missing bmp file\n");
    return;
  }

  char filename[1024];

  if (pixels == NULL)
    sprintf(filename, "%s_%d_buffer.bmp", fn, step);
  else
    sprintf(filename, "%s_%d_pixels.bmp", fn, step);

  int width = tile_w;
  int height = tile_h;

  int extrabytes = 4 - ((width * 3) % 4);
  if (extrabytes == 4)
    extrabytes = 0;

  int paddedsize = ((width * 3) + extrabytes) * height;

  unsigned int headers[13];
  headers[0] = paddedsize + 54;
  headers[1] = 0;
  headers[2] = 54;
  headers[3] = 40;
  headers[4] = width;
  headers[5] = height;

  headers[7] = 0;
  headers[8] = paddedsize;
  headers[9] = 0;
  headers[10] = 0;
  headers[11] = 0;
  headers[12] = 0;

  FILE *outfile = fopen(filename, "wb+");

  fprintf(outfile, "BM");

  for (int n = 0; n <= 5; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  fprintf(outfile, "%c", 1);
  fprintf(outfile, "%c", 0);
  fprintf(outfile, "%c", 24);
  fprintf(outfile, "%c", 0);

  for (int n = 7; n <= 12; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  for (int y = tile_y; y <= height + tile_y - 1; y++) {
    for (int x = tile_x; x <= width + tile_x - 1; x++) {

      unsigned char byte_result[4];

      if (pixels == NULL) {
        util_film_convert_to_byte(
            (float *)buffer, 1.0f / end_samples, x, y, offset, stride, pass_stride, byte_result);
      }
      else {
        size_t index = (offset + x + y * stride) * 4;
        unsigned short *pixels_half = (unsigned short *)pixels + index;
        byte_result[0] = util_half_to_byte(pixels_half[0]);
        byte_result[1] = util_half_to_byte(pixels_half[1]);
        byte_result[2] = util_half_to_byte(pixels_half[2]);
        byte_result[3] = util_half_to_byte(pixels_half[3]);
      }

      fprintf(outfile, "%c", byte_result[2]);
      fprintf(outfile, "%c", byte_result[1]);
      fprintf(outfile, "%c", byte_result[0]);
    }
    if (extrabytes) {
      for (int n = 1; n <= extrabytes; n++) {
        fprintf(outfile, "%c", 0);
      }
    }
  }

  fclose(outfile);

  printf("Saved: %s\n", filename);
}

void util_save_yuv(
    unsigned char *dst_y, unsigned char *dst_u, unsigned char *dst_v, int width, int height)
{
  const char *filename = std::getenv("YUV_OUTPUT");
  if (filename == NULL) {
    return;
  }

  printf("writing data to %s\n", filename);

  FILE *f = fopen(filename, "w+");
  fwrite(dst_y, 1, width * height, f);
  fwrite(dst_u, 1, width * height / 4, f);
  fwrite(dst_v, 1, width * height / 4, f);
  fclose(f);
}

int util_get_int_from_env_array(const char *env_value, int id)
{
  if (env_value == NULL)
    return 0;

  std::stringstream test(env_value);
  std::string segment;

  int i = 0;
  while (std::getline(test, segment, ';')) {

    if (i == id)
      return atoi(segment.c_str());

    i++;
  }

  return 0;
}

int util_get_count_from_env_array(const char *env_value)
{
  if (env_value == NULL)
    return 16;

  std::stringstream test(env_value);
  std::string segment;

  int i = 0;
  while (std::getline(test, segment, ',')) {
    i++;
  }

  if (i == 0)
    i = 16;

  return i;
}

float util_max(float a, float b)
{
  return (a > b) ? a : b;
}

float util_min(float a, float b)
{
  return (a < b) ? a : b;
}

float util_clamp(float a, float mn, float mx)
{
  return util_min(util_max(a, mn), mx);
}

void util_rgb_to_yuv_i420_stereo(unsigned char *destination,
                                 unsigned char *source,
                                 int tile_h,
                                 int tile_w)
{
  unsigned char *dst_y = destination;
  unsigned char *dst_u = destination + tile_w * tile_h;
  unsigned char *dst_v = destination + tile_w * tile_h + tile_w * tile_h / 4;

#pragma omp parallel for
  for (int y = 0; y < tile_h; y++) {
    for (int x = 0; x < tile_w; x++) {

      int index_src = x + y * tile_w / 2;

      if (x >= tile_w / 2) {
        index_src = (x / 2 + y * tile_w / 2) + (tile_h * tile_w / 2);
      }

      unsigned char r = source[index_src * 4 + 0];
      unsigned char g = source[index_src * 4 + 1];
      unsigned char b = source[index_src * 4 + 2];

      // Y
      int index_y = x + y * tile_w;
      dst_y[index_y] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

      // U
      if (x % 2 == 0 && y % 2 == 0) {
        int index_u = (x / 2) + (y / 2) * (tile_w / 2);
        dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
      }

      // V
      if (x % 2 == 0 && y % 2 == 0) {
        int index_v = (x / 2) + (y / 2) * (tile_w / 2);
        dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
      }
    }
  }
}

void util_rgb_to_yuv_i420(unsigned char *destination,
                          unsigned char *source,
                          int tile_h,
                          int tile_w)
{
  unsigned char *dst_y = destination;
  unsigned char *dst_u = destination + tile_w * tile_h;
  unsigned char *dst_v = destination + tile_w * tile_h + tile_w * tile_h / 4;

#pragma omp parallel for
  for (int y = 0; y < tile_h; y++) {
    for (int x = 0; x < tile_w; x++) {

      int index_src = x + y * tile_w;

      unsigned char r = source[index_src * 4 + 0];
      unsigned char g = source[index_src * 4 + 1];
      unsigned char b = source[index_src * 4 + 2];

      // Y
      int index_y = x + y * tile_w;
      dst_y[index_y] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

      // U
      if (x % 2 == 0 && y % 2 == 0) {
        int index_u = (x / 2) + (y / 2) * (tile_w / 2);
        dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
      }

      // V
      if (x % 2 == 0 && y % 2 == 0) {
        int index_v = (x / 2) + (y / 2) * (tile_w / 2);
        dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
      }
    }
  }
}

void util_yuv_i420_to_rgba(unsigned char *destination,
                           unsigned char *source,
                           int tile_h,
                           int tile_w)
{

  unsigned char *src_y = source;
  unsigned char *src_u = source + tile_w * tile_h;
  unsigned char *src_v = source + tile_w * tile_h + tile_w * tile_h / 4;

  size_t image_size = tile_h * tile_w;

#pragma omp parallel for
  for (int y = 0; y < tile_h; y++) {
    for (int x = 0; x < tile_w; x++) {

      int index_y = x + y * tile_w;
      int C = src_y[index_y] - 16;

      int index_u = (x / 2) + (y / 2) * (tile_w / 2);
      int D = src_u[index_u] - 128;

      int index_v = (x / 2) + (y / 2) * (tile_w / 2);
      int E = src_v[index_v] - 128;

      unsigned char R = util_clamp((298 * C + 409 * E + 128) >> 8, 0, 255);
      unsigned char G = util_clamp((298 * C - 100 * D - 208 * E + 128) >> 8, 0, 255);
      unsigned char B = util_clamp((298 * C + 516 * D + 128) >> 8, 0, 255);

      int index_dst = x + y * tile_w;

      destination[index_dst * 4 + 0] = R;
      destination[index_dst * 4 + 1] = G;
      destination[index_dst * 4 + 2] = B;
      destination[index_dst * 4 + 3] = 255;
    }
  }
}

std::vector<char> util_rgb_to_xor_rle_data;
std::vector<char> util_rgb_to_xor_rle_temp;
std::vector<char> util_rgb_to_xor_rle_temp2;

char *util_rgb_to_xor_rle(char *source, int tile_h, int tile_w, size_t &res_size)
{
  //    if (util_rgb_to_xor_rle_data.size() != (size_t)tile_w * tile_h * 2 * 3) {
  //        util_rgb_to_xor_rle_data.resize((size_t)tile_w * tile_h * 2 * 3);
  //        util_rgb_to_xor_rle_temp.resize((size_t)tile_w * tile_h * 4);
  //
  //        size_t size = (size_t)tile_w * tile_h;
  //        char* destination_r = &util_rgb_to_xor_rle_data[0];
  //        char* destination_g = destination_r + (size_t)tile_w * tile_h;
  //        char* destination_b = destination_g + (size_t)tile_w * tile_h;
  //        //unsigned char* destination_a = destination_b + tile_w * tile_h;
  //
  //#pragma omp parallel for
  //        for (int i = 0; i < size; i++) {
  //            destination_r[i] = source[i * 4 + 0];
  //            destination_g[i] = source[i * 4 + 1];
  //            destination_b[i] = source[i * 4 + 2];
  //            //destination_a[i] = source_key[i * 4 + 3];
  //        }
  //
  //        memcpy(&util_rgb_to_xor_rle_temp[0], source, (size_t)tile_w * tile_h * 4);
  //
  //        printf("util_rgb_to_xor_rle: ----key----: %lld\n", size * 3);
  //        //res_size = size * 3;
  //    }
  //    else {
  //
  //        size_t size = (size_t)tile_w * tile_h;
  //        char* destination_r = &util_rgb_to_xor_rle_data[0];
  //        char* destination_g = destination_r + (size_t)tile_w * tile_h;
  //        char* destination_b = destination_g + (size_t)tile_w * tile_h;
  //        //unsigned char* destination_a = destination_b + tile_w * tile_h;
  //
  //        unsigned char* source_key = (unsigned char*)&util_rgb_to_xor_rle_temp[0];
  //
  //#pragma omp parallel for
  //        for (int i = 0; i < size; i++) {
  //            destination_r[i] = (source_key[i * 4 + 0] != source[i * 4 + 0]) ? source[i * 4 + 0]
  //            : 0; destination_g[i] = (source_key[i * 4 + 1] != source[i * 4 + 1]) ? source[i * 4
  //            + 1] : 0; destination_b[i] = (source_key[i * 4 + 2] != source[i * 4 + 2]) ?
  //            source[i * 4 + 2] : 0;
  //            //destination_a[i] = source_key[i * 4 + 3];
  //        }
  //
  //        memcpy(&util_rgb_to_xor_rle_temp[0], source, (size_t)tile_w* tile_h * 4);
  //        printf("util_rgb_to_xor_rle: %lld\n", size * 3);
  //        //res_size = size * 3;
  //    }

  int rle_multi = 2;  // key+value
  size_t xor_rle_data_size = ((size_t)tile_w * tile_h * 3) * rle_multi + sizeof(size_t) * 4;

  if (util_rgb_to_xor_rle_data.size() != xor_rle_data_size) {
    util_rgb_to_xor_rle_data.resize(xor_rle_data_size);
  }

  if (util_rgb_to_xor_rle_temp.size() != xor_rle_data_size) {
    util_rgb_to_xor_rle_temp.resize(xor_rle_data_size);
  }

  if (util_rgb_to_xor_rle_temp2.size() != xor_rle_data_size) {
    util_rgb_to_xor_rle_temp2.resize(xor_rle_data_size);
  }

  // convert to YUV
  unsigned char *dst_y = (unsigned char *)&util_rgb_to_xor_rle_temp[0] + sizeof(size_t) * 4;
  unsigned char *dst_u = dst_y + (size_t)tile_w * tile_h;
  unsigned char *dst_v = dst_y + (size_t)tile_w * tile_h * 2;

#pragma omp parallel for
  for (int y = 0; y < tile_h; y++) {
    for (int x = 0; x < tile_w; x++) {

      int index_src = x + y * tile_w;

      unsigned char r = source[index_src * 4 + 0];
      unsigned char g = source[index_src * 4 + 1];
      unsigned char b = source[index_src * 4 + 2];

      // Y
      int index_y = x + y * tile_w;
      dst_y[index_y] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

      //// U
      // if (x % 2 == 0 && y % 2 == 0) {
      //    int index_u = (x / 2) + (y / 2) * (tile_w / 2);
      //    dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
      //}

      //// V
      // if (x % 2 == 0 && y % 2 == 0) {
      //    int index_v = (x / 2) + (y / 2) * (tile_w / 2);
      //    dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
      //}

      // U
      int index_u = x + y * tile_w;
      dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;

      // V
      int index_v = x + y * tile_w;
      dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
    }
  }

  unsigned char *dst_y2 = (unsigned char *)&util_rgb_to_xor_rle_temp2[0] + sizeof(size_t) * 4;
  unsigned char *dst_u2 = dst_y + (size_t)tile_w * tile_h * rle_multi;
  unsigned char *dst_v2 = dst_y + (size_t)tile_w * tile_h * rle_multi * 2;

  size_t image_size = (size_t)tile_w * tile_h;

  size_t dst_y2_id = 0;
  size_t dst_u2_id = 0;
  size_t dst_v2_id = 0;

  size_t dst_y2_rle_count = 0;
  size_t dst_u2_rle_count = 0;
  size_t dst_v2_rle_count = 0;

  // size_t dst_y2_rle_value = dst_y[0];
  // size_t dst_u2_rle_value = dst_u[0];
  // size_t dst_v2_rle_value = dst_v[0];

  //#pragma omp parallel for
  for (int i = 0; i < image_size; i++) {
    dst_y2_rle_count++;
    dst_u2_rle_count++;
    dst_v2_rle_count++;

    if (i == image_size - 1 || dst_y2_rle_count == 255 || dst_y[i] != dst_y[i + 1]) {
      dst_y2[dst_y2_id++] = dst_y2_rle_count;
      dst_y2[dst_y2_id++] = dst_y[i];

      dst_y2_rle_count = 0;
    }

    if (i == image_size - 1 || dst_u2_rle_count == 255 || dst_u[i] != dst_u[i + 1]) {
      dst_u2[dst_u2_id++] = dst_u2_rle_count;
      dst_u2[dst_u2_id++] = dst_u[i];

      dst_u2_rle_count = 0;
    }

    if (i == image_size - 1 || dst_v2_rle_count == 255 || dst_v[i] != dst_v[i + 1]) {
      dst_v2[dst_v2_id++] = dst_v2_rle_count;
      dst_v2[dst_v2_id++] = dst_v[i];

      dst_v2_rle_count = 0;
    }
  }

  unsigned char *dst = (unsigned char *)&util_rgb_to_xor_rle_data[0] + sizeof(size_t) * 4;
  memcpy(dst, dst_y2, dst_y2_id);
  memcpy(dst + dst_y2_id, dst_u2, dst_u2_id);
  memcpy(dst + dst_y2_id + dst_u2_id, dst_v2, dst_v2_id);

  res_size = dst_y2_id + dst_u2_id + dst_v2_id + sizeof(size_t) * 4;
  size_t *dst_size = (size_t *)&util_rgb_to_xor_rle_data[0];
  dst_size[0] = res_size;
  dst_size[1] = dst_y2_id;
  dst_size[2] = dst_u2_id;
  dst_size[3] = dst_v2_id;

  return &util_rgb_to_xor_rle_data[0];
}

std::vector<char> util_xor_rle_to_rgb_data;
std::vector<char> util_xor_rle_to_rgb_temp;
// std::vector<char> util_xor_rle_to_rgb_temp2;

char *util_xor_rle_to_rgb(char *source, int tile_h, int tile_w, size_t res_size)
{
  //    if (util_xor_rle_to_rgb_data.size() != (size_t)tile_w * tile_h * 2 * 4) {
  //        util_xor_rle_to_rgb_data.resize((size_t)tile_w * tile_h * 2 * 4);
  //        //util_xor_rle_to_rgb_temp.resize((size_t)tile_w * tile_h * 2 * 3);
  //
  //        size_t size = (size_t)tile_w * tile_h;
  //        char* destination = &util_xor_rle_to_rgb_data[0];
  //
  //        char* s_r = source;
  //        char* s_g = s_r + (size_t)tile_w * tile_h;
  //        char* s_b = s_g + (size_t)tile_w * tile_h;
  //
  //#pragma omp parallel for
  //        for (int i = 0; i < size; i++) {
  //            destination[i * 4 + 0] = s_r[i];
  //            destination[i * 4 + 1] = s_g[i];
  //            destination[i * 4 + 2] = s_b[i];
  //            destination[i * 4 + 3] = 255;
  //            //destination_a[i] = source_key[i * 4 + 3];
  //        }
  //
  //        printf("util_xor_rle_to_rgb: ----key----: %lld\n", size * 4);
  //    }
  //    else {
  //        size_t size = (size_t)tile_w * tile_h;
  //        char* destination = &util_xor_rle_to_rgb_data[0];
  //
  //        char* s_r = source;
  //        char* s_g = s_r + (size_t)tile_w * tile_h;
  //        char* s_b = s_g + (size_t)tile_w * tile_h;
  //
  //#pragma omp parallel for
  //        for (int i = 0; i < size; i++) {
  //            destination[i * 4 + 0] = (s_r[i] != 0) ? s_r[i] : destination[i * 4 + 0];
  //            destination[i * 4 + 1] = (s_g[i] != 0) ? s_g[i] : destination[i * 4 + 1];
  //            destination[i * 4 + 2] = (s_b[i] != 0) ? s_b[i] : destination[i * 4 + 2];
  //        }
  //
  //        printf("util_rgb_to_xor_rle: ----key----: %lld\n", size * 3);
  //    }

  // int rle_multi = 2; // key+value
  size_t xor_rle_data_size = (size_t)tile_w * tile_h * 4;

  if (util_xor_rle_to_rgb_data.size() != xor_rle_data_size) {
    util_xor_rle_to_rgb_data.resize(xor_rle_data_size);
  }

  int rle_multi = 2;  // key+value
  size_t xor_rle_temp_size = ((size_t)tile_w * tile_h * 3) * rle_multi + sizeof(size_t) * 4;

  if (util_xor_rle_to_rgb_temp.size() != xor_rle_temp_size) {
    util_xor_rle_to_rgb_temp.resize(xor_rle_temp_size);
  }

  // if (util_xor_rle_to_rgb_temp2.size() != xor_rle_temp_size) {
  //    util_xor_rle_to_rgb_temp2.resize(xor_rle_temp_size);
  //}

  size_t *src_size = (size_t *)source;
  size_t src_y2_id = src_size[0];
  size_t src_u2_id = src_size[1];
  size_t src_v2_id = src_size[2];

  unsigned char *src_y2 = (unsigned char *)source + sizeof(size_t) * 3;
  unsigned char *src_u2 = src_y2 + src_y2_id;
  unsigned char *src_v2 = src_y2 + src_y2_id + src_u2_id;

  unsigned char *src_y = (unsigned char *)&util_xor_rle_to_rgb_temp[0] + sizeof(size_t) * 3;
  unsigned char *src_u = src_y + (size_t)tile_w * tile_h;
  unsigned char *src_v = src_y + (size_t)tile_w * tile_h * 2;

  size_t src_y_id = 0;
  size_t src_u_id = 0;
  size_t src_v_id = 0;

  for (int i = 0; i < src_y2_id; i += 2) {
    for (int j = 0; j < src_y2[i]; j++) {
      src_y[src_y_id++] = src_y2[i + 1];
    }
  }

  for (int i = 0; i < src_u2_id; i += 2) {
    for (int j = 0; j < src_u2[i]; j++) {
      src_u[src_u_id++] = src_u2[i + 1];
    }
  }

  for (int i = 0; i < src_v2_id; i += 2) {
    for (int j = 0; j < src_v2[i]; j++) {
      src_v[src_v_id++] = src_v2[i + 1];
    }
  }

  char *destination = &util_xor_rle_to_rgb_data[0];

  size_t image_size = (size_t)tile_w * tile_h;

#pragma omp parallel for
  for (int i = 0; i < image_size; i++) {
    int C = src_y[i] - 16;
    int D = src_u[i] - 128;
    int E = src_v[i] - 128;

    unsigned char R = util_clamp((298 * C + 409 * E + 128) >> 8, 0, 255);
    unsigned char G = util_clamp((298 * C - 100 * D - 208 * E + 128) >> 8, 0, 255);
    unsigned char B = util_clamp((298 * C + 516 * D + 128) >> 8, 0, 255);

    destination[i * 4 + 0] = R;
    destination[i * 4 + 1] = G;
    destination[i * 4 + 2] = B;
    destination[i * 4 + 3] = 255;
  }

  return &util_xor_rle_to_rgb_data[0];
}

void util_convert_to_xor_rle()
{
}

/* ***** Jenkins Lookup3 Hash Functions ***** */

/* Source: http://burtleburtle.net/bob/c/lookup3.c */

#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

#define mix(a, b, c) \
  { \
    a -= c; \
    a ^= rot(c, 4); \
    c += b; \
    b -= a; \
    b ^= rot(a, 6); \
    a += c; \
    c -= b; \
    c ^= rot(b, 8); \
    b += a; \
    a -= c; \
    a ^= rot(c, 16); \
    c += b; \
    b -= a; \
    b ^= rot(a, 19); \
    a += c; \
    c -= b; \
    c ^= rot(b, 4); \
    b += a; \
  } \
  ((void)0)

#define final(a, b, c) \
  { \
    c ^= b; \
    c -= rot(b, 14); \
    a ^= c; \
    a -= rot(c, 11); \
    b ^= a; \
    b -= rot(a, 25); \
    c ^= b; \
    c -= rot(b, 16); \
    a ^= c; \
    a -= rot(c, 4); \
    b ^= a; \
    b -= rot(a, 14); \
    c ^= b; \
    c -= rot(b, 24); \
  } \
  ((void)0)

unsigned int util_hash_uint2(unsigned int kx, unsigned int ky)
{
  unsigned int a, b, c;
  a = b = c = 0xdeadbeef + (2 << 2) + 13;

  b += ky;
  a += kx;
  final(a, b, c);

  return c;
}