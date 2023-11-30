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

#include "kernel_file.h"

#include "../../../util/types.h"
#include <cstdio>
#include <omp.h>
#include <string.h>
#include <vector>

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#  include <io.h>
#  include <sys/types.h>
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#endif

#define UCHAR4 uchar4

// CCL_NAMESPACE_BEGIN

#ifdef _WIN32
// https://github.com/m-labs/uclibc-lm32/blob/master/utils/mmap-windows.c

#  define PROT_READ 0x1
#  define PROT_WRITE 0x2
/* This flag is only available in WinXP+ */
#  ifdef FILE_MAP_EXECUTE
#    define PROT_EXEC 0x4
#  else
#    define PROT_EXEC 0x0
#    define FILE_MAP_EXECUTE 0
#  endif

#  define MAP_SHARED 0x01
#  define MAP_PRIVATE 0x02
#  define MAP_ANONYMOUS 0x20
#  define MAP_ANON MAP_ANONYMOUS
#  define MAP_FAILED ((void *)-1)

#  ifdef __USE_FILE_OFFSET64
#    define DWORD_HI(x) (x >> 32)
#    define DWORD_LO(x) ((x)&0xffffffff)
#  else
#    define DWORD_HI(x) (0)
#    define DWORD_LO(x) (x)
#  endif

static void *mmap(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
  if (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))
    return MAP_FAILED;
  if (fd == -1) {
    if (!(flags & MAP_ANON) || offset)
      return MAP_FAILED;
  }
  else if (flags & MAP_ANON)
    return MAP_FAILED;

  DWORD flProtect;
  if (prot & PROT_WRITE) {
    if (prot & PROT_EXEC)
      flProtect = PAGE_EXECUTE_READWRITE;
    else
      flProtect = PAGE_READWRITE;
  }
  else if (prot & PROT_EXEC) {
    if (prot & PROT_READ)
      flProtect = PAGE_EXECUTE_READ;
    else if (prot & PROT_EXEC)
      flProtect = PAGE_EXECUTE;
  }
  else
    flProtect = PAGE_READONLY;

  off_t end = length + offset;
  HANDLE mmap_fd, h;
  if (fd == -1)
    mmap_fd = INVALID_HANDLE_VALUE;
  else
    mmap_fd = (HANDLE)_get_osfhandle(fd);
  h = CreateFileMapping(mmap_fd, NULL, flProtect, DWORD_HI(end), DWORD_LO(end), NULL);
  if (h == NULL)
    return MAP_FAILED;

  DWORD dwDesiredAccess;
  if (prot & PROT_WRITE)
    dwDesiredAccess = FILE_MAP_WRITE;
  else
    dwDesiredAccess = FILE_MAP_READ;
  if (prot & PROT_EXEC)
    dwDesiredAccess |= FILE_MAP_EXECUTE;
  if (flags & MAP_PRIVATE)
    dwDesiredAccess |= FILE_MAP_COPY;
  void *ret = MapViewOfFile(h, dwDesiredAccess, DWORD_HI(offset), DWORD_LO(offset), length);
  if (ret == NULL) {
    CloseHandle(h);
    ret = MAP_FAILED;
  }
  return ret;
}

static bool munmap(void *addr, size_t length)
{
  return UnmapViewOfFile(addr);
  /* ruh-ro, we leaked handle from CreateFileMapping() ... */
}
#endif

namespace cyclesphi {
namespace kernel {
namespace file {

char *g_client_kernel_global = 0;
size_t g_client_kernel_global_size = 0;
size_t g_client_kernel_global_id = 0;

struct FileCache {
  client_kernel_struct data;
  char *mem;
  size_t size;
  FileCache(client_kernel_struct _data, char *_mem = NULL, size_t _size = 0)
      : data(_data), mem(_mem), size(_size)
  {
  }
  ~FileCache()
  {
  }
};

bool g_use_delete_from_cache = false;
std::vector<FileCache> g_cache_map;

bool init_kernel_global();
void get_kernel_data(client_kernel_struct *data, int client_tag);
void send_kernel_data(client_kernel_struct *data);

void unmap(char *mem_buffer)
{
  // if (!munmap(mem_buffer, g_client_kernel_global_size)) {
  if (munmap(mem_buffer, g_client_kernel_global_size)) {
    perror("Could not unmap file");
    exit(1);
  }

  // printf("Successfully unmapped file %s.\n", mem_buffer);
}

char *map(const char *path)
{
  int fd = open(path, O_RDWR, S_IREAD);

  if (fd < 0) {
    perror("Could not open file for memory mapping");
    exit(1);
  }

  struct stat stat_buf;
  int rc = fstat(fd, &stat_buf);

  g_client_kernel_global_size = (rc == 0) ? stat_buf.st_size : -1;
  g_client_kernel_global_id = 0;

  char *mem_buffer = (char *)mmap(
      NULL, g_client_kernel_global_size, PROT_READ, MAP_PRIVATE, fd, 0);

  if (mem_buffer == MAP_FAILED) {
    perror("Could not memory map file");
    exit(1);
  }

  // printf("Successfully mapped file: %s.\n", path);

  return mem_buffer;
}

void close_kernelglobal()
{
  if (g_client_kernel_global != 0) {

#ifdef WITH_CLIENT_FILE_MMAP
    unmap(g_client_kernel_global);
#else
    fclose((FILE *)g_client_kernel_global);
#endif

    g_client_kernel_global = 0;
  }
}

bool init_kernel_global()
{
  if (g_client_kernel_global == 0) {
    const char *name = getenv("CLIENT_FILE_KERNEL_GLOBAL");

    if (name == 0) {
      printf("missing CLIENT_FILE_KERNEL_GLOBAL file\n");
      // return false;
      exit(-1);
    }
    else {
      printf("CLIENT_FILE_KERNEL_GLOBAL file: %s\n", name);
    }

#if defined(BLENDER_CLIENT)

#  ifdef WITH_CLIENT_FILE_MMAP
    g_client_kernel_global = (char *)map(name);
#  else
    g_client_kernel_global = (char *)fopen(name, "rb");
#  endif

#else
    g_client_kernel_global = (char *)fopen(name, "wb");
#endif
    // printf("CLIENT_FILE_KERNEL_GLOBAL: %s\n", name);
    if (!g_client_kernel_global) {
      printf("cannot open the file %s\n", name);
      exit(-1);
    }
  }

  return true;
}

void write_data_kernelglobal(void *data, size_t size)
{
  if (init_kernel_global() && is_preprocessing())
    fwrite(data, size, 1, (FILE *)g_client_kernel_global);
}

bool read_data_kernelglobal(void *data, size_t size)
{
  if (init_kernel_global()) {
    size_t readed = 0;
#ifdef __MIC__
    const size_t step = 1024L * 1024L * 2L;  // 128L;
    if (size < step) {
#endif

      // double t = omp_get_wtime();

#ifdef WITH_CLIENT_FILE_MMAP
      char *cdata = (char *)data;

      // very slow
#  if defined(WITH_CUDA_STATv2) || defined(WITH_HIP_STATv2) || \
      defined(WITH_CLIENT_CUDA_CPU_STAT2v2)

#    pragma omp parallel for  // schedule(dynamic, 1000000)
      for (size_t i = 0; i < size; i++) {
        cdata[i] = g_client_kernel_global[i + g_client_kernel_global_id];
      }

#  else

// bug which causes crashes
//#if 0
#    pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int tnum = omp_get_num_threads();

        size_t tsize = size / tnum;
        size_t offset = tid * tsize;

        if (tid == tnum - 1)
          tsize = size - tid * tsize;

        size_t chunk = 1L * 1024L * 1024L;
        for (size_t i = 0; i < tsize; i += chunk) {
          if (((i + chunk) > tsize) && (tsize - i) < 0)
            break;
          memcpy(cdata + offset + i,
                 g_client_kernel_global + offset + i + g_client_kernel_global_id,
                 ((i + chunk) > tsize) ? tsize - i : chunk);
        }
      }
#  endif

      g_client_kernel_global_id += size;

#else
    readed += fread((char *)data, 1, size, (FILE *)g_client_kernel_global);
#endif

      // printf("read_data_kernelglobal: %f\n", omp_get_wtime() - t);

#ifdef __MIC__
    }
    else {
      size_t offset = 0;
      while (offset + step < size) {
        readed += fread((char *)data + offset, step, 1, (FILE *)g_client_kernel_global);
        offset += step;

        if (readed == 0)
          break;
      }

      if (size - offset > 0)
        readed += fread((char *)data + offset, size - offset, 1, (FILE *)g_client_kernel_global);
    }
#endif

    return readed != 0;
  }

  return false;
}

/////////////////////////////////////////////////////

void get_kernel_data(client_kernel_struct *data, int client_tag)
{
  memset(data, 0, sizeof(client_kernel_struct));
  data->client_tag = client_tag;
}

bool delete_from_cache(client_kernel_struct &data)
{
  bool req_add = true;

  int i = 0;
  while (i < g_cache_map.size()) {
    if (data.client_tag == CLIENT_TAG_CYCLES_const_copy) {
      if (g_cache_map[i].data.client_tag == CLIENT_TAG_CYCLES_const_copy) {
        g_cache_map.erase(g_cache_map.begin() + i);
        continue;
      }
    }

    if (data.client_tag == CLIENT_TAG_CYCLES_tex_free) {
      if (data.client_mem_data.mem != NULL &&
          (g_cache_map[i].data.client_tex_copy_data.mem == data.client_mem_data.mem ||
           g_cache_map[i].data.client_mem_data.mem == data.client_mem_data.mem)) {
        g_cache_map.erase(g_cache_map.begin() + i);
        req_add = false;
        continue;
      }
    }

    if (data.client_tag == CLIENT_TAG_CYCLES_mem_free) {
      if (data.client_mem_data.mem != NULL &&
          g_cache_map[i].data.client_mem_data.mem == data.client_mem_data.mem) {
        g_cache_map.erase(g_cache_map.begin() + i);
        req_add = false;
        continue;
      }
    }
    i++;
  }

  return req_add;
}

void send_kernel_data()
{

  for (int i = 0; i < g_cache_map.size(); i++) {
    write_data_kernelglobal(&g_cache_map[i].data, sizeof(client_kernel_struct));

    if (g_cache_map[i].mem != NULL)
      write_data_kernelglobal(g_cache_map[i].mem, g_cache_map[i].size);
  }
}

void send_to_cache(client_kernel_struct &data, void *mem = NULL, size_t size = 0)
{
  write_data_kernelglobal(&data, sizeof(client_kernel_struct));

  if (mem != NULL)
    write_data_kernelglobal(mem, size);

  if (data.client_tag == CLIENT_TAG_CYCLES_free_kg) {

    close_kernelglobal();
  }
}

int count_devices()
{
  return 1;
}

bool read_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size)
{
  const char *filename = getenv("CLIENT_FILE_CYCLES_BUFFER");
  if (filename == 0) {
    printf("missing CLIENT_FILE_CYCLES_BUFFER file\n");
    return false;
  }

  FILE *file = fopen(filename, "rb");

  fread((char *)samples, sizeof(int), 1, file);
  fread(buffer + offset, size, 1, file);

  fclose(file);

  printf("Read: %s\n", filename);

  return true;
}

bool read_cycles_data(const char *filename, char *buffer, size_t offset, size_t size)
{
  // const char* filename = getenv("CLIENT_FILE_CYCLES_BUFFER");
  if (filename == 0) {
    printf("missing CLIENT_FILE_CYCLES_BUFFER file\n");
    return false;
  }

  FILE *file = fopen(filename, "rb");

  // fread((char*)samples, sizeof(int), 1, file);
  fread(buffer + offset, size, 1, file);

  fclose(file);

  printf("Read: %s\n", filename);

  return true;
}

bool write_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size)
{
  const char *filename = getenv("CLIENT_FILE_CYCLES_BUFFER");
  if (filename == 0) {
    printf("missing CLIENT_FILE_CYCLES_BUFFER file\n");
    return false;
  }

  FILE *file = fopen(filename, "wb");

  fwrite((char *)samples, sizeof(int), 1, file);
  fwrite(buffer + offset, size, 1, file);

  fclose(file);

  printf("Write: %s\n", filename);

  return true;
}

bool write_cycles_data(const char *filename, char *buffer, size_t offset, size_t size)
{
  // const char* filename = getenv("CLIENT_FILE_CYCLES_BUFFER");
  if (filename == 0) {
    printf("missing CLIENT_FILE_CYCLES_BUFFER file\n");
    return false;
  }

  FILE *file = fopen(filename, "wb");

  // fwrite((char*)samples, sizeof(int), 1, file);
  fwrite(buffer + offset, size, 1, file);

  fclose(file);

  printf("Write: %s\n", filename);

  return true;
}

////////////////////////////////////////////////////////////////////////////

bool is_preprocessing()
{
  const char *action = getenv("CLIENT_FILE_ACTION");
  if (action == 0) {
    printf("missing CLIENT_FILE_ACTION file\n");
    return false;
  }

  return !strcmp(action, "PRE");
}

bool is_postprocessing()
{
  const char *action = getenv("CLIENT_FILE_ACTION");
  if (action == 0) {
    printf("missing CLIENT_FILE_ACTION file\n");
    return false;
  }

  return !strcmp(action, "POST");
}

int get_additional_samples()
{
  const char *s_samples = getenv("CLIENT_FILE_ADDITIONAL_SAMPLES");
  if (s_samples == 0) {
    return 0;
  }

  return atoi(s_samples);
}

void receive_path_trace_buffer(int offset,
                               int stride,
                               int tile_x,
                               int tile_y,
                               int tile_h,
                               int tile_w,
                               size_t pass_stride_sizeof,
                               char *tile_buffer)
{
  read_data_kernelglobal((char *)tile_buffer +
                             (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                         tile_w * tile_h * pass_stride_sizeof);
}

void const_copy(const char *name, char *host_bin, size_t size, DEVICE_PTR host_ptr)
{
  if (strcmp(name, "data") == 0) {
    client_kernel_struct data;
    get_kernel_data(&data, CLIENT_TAG_CYCLES_const_copy);

    strcpy(data.client_const_copy_data.name, name);
    data.client_const_copy_data.host = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)host_bin;
    data.client_const_copy_data.size = size;
    data.client_const_copy_data.read_data = true;

    send_to_cache(data, host_bin, size);
  }
}

void load_textures(size_t size, client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_load_textures);

  data.client_load_textures_data.texture_info_size = size;

  send_to_cache(data);
}

void tex_copy(const char *name,
              void *mem,
              size_t data_size,
              size_t mem_size,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{

  // printf("tex_copy: %s, %zu, %zu\n", name, data_size, mem_size);

  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_tex_copy);

  strcpy(data.client_tex_copy_data.name, name);
  data.client_tex_copy_data.mem = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)mem;
  data.client_tex_copy_data.data_size = data_size;
  data.client_tex_copy_data.mem_size = mem_size;

  send_to_cache(data, (char *)mem, mem_size);
}

void alloc_kg(client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_alloc_kg);

  send_to_cache(data);
}

void free_kg(client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_free_kg);

  send_to_cache(data);
}

void mem_alloc(const char *name,
               DEVICE_PTR mem,
               size_t memSize,
               DEVICE_PTR host_ptr,
               client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_mem_alloc);

  strcpy(data.client_mem_data.name, name);
  data.client_mem_data.mem = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;

  send_to_cache(data);
}

void mem_alloc_sub_ptr(const char *name,
                       DEVICE_PTR mem,
                       size_t offset,
                       DEVICE_PTR mem_sub,
                       client_kernel_struct *_client_data)
{
}

void mem_copy_to(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 size_t offset,
                 DEVICE_PTR host_ptr,
                 client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_mem_copy_to);

  strcpy(data.client_mem_data.name, name);
  data.client_mem_data.mem = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;
  data.client_mem_data.offset = offset;

  send_to_cache(data, (char *)mem, memSize);
}

void mem_zero(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              size_t offset,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_mem_zero);

  strcpy(data.client_mem_data.name, name);
  data.client_mem_data.mem = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;
  data.client_mem_data.offset = offset;

  send_to_cache(data);
}

void mem_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_mem_free);

  strcpy(data.client_mem_data.name, name);
  data.client_mem_data.mem = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;

  send_to_cache(data);
}

void tex_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_tex_free);

  strcpy(data.client_mem_data.name, name);
  data.client_mem_data.mem = (host_ptr != 0) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;

  send_to_cache(data);
}

void tex_info(char *mem,
              char *mem_data,
              size_t size,
              const char *name,
              int data_type,
              int data_elements,
              int interpolation,
              int extension,
              size_t data_width,
              size_t data_height,
              size_t data_depth)
{
  // mem_alloc((device_ptr) mem, size, NULL);
  // mem_copy_to((device_ptr) mem, size, 0, NULL);

  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_tex_info);

  strcpy(data.client_tex_info_data.name, name);

  data.client_tex_info_data.data_type = data_type;
  data.client_tex_info_data.data_elements = data_elements;
  data.client_tex_info_data.interpolation = interpolation;

  data.client_tex_info_data.extension = extension;
  data.client_tex_info_data.data_width = data_width;
  data.client_tex_info_data.data_height = data_height;
  data.client_tex_info_data.data_depth = data_depth;

  data.client_tex_info_data.mem = (DEVICE_PTR)mem;
  data.client_tex_info_data.size = size;

  send_to_cache(data, (char *)mem_data, size);
}

void convert_rgb_to_half(unsigned short *destination,
                         unsigned char *source,
                         int tile_h,
                         int tile_w)
{
}
void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
}
void set_kernel_globals(char *kg)
{
}

#ifdef WITH_CLIENT_OPTIX
void build_optix_bvh(int operation, char *build_input, size_t build_size, int num_motion_steps)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_build_bvh);

  data.client_build_bvh_data.operation = operation;

  data.client_build_bvh_data.build_input = (DEVICE_PTR)build_input;
  data.client_build_bvh_data.build_size = build_size;

  data.client_build_bvh_data.num_motion_steps = num_motion_steps;

  send_to_cache(data, (char *)build_input, build_size);
}
#endif
/////////////////////////////////////////////////////////////////////
void path_trace(char *buffer,
                char *pixels,
                int start_sample,
                int num_samples,
                int sample_offset,
                int tile_x,
                int tile_y,
                int offset,
                int stride,
                int tile_h,
                int tile_w,
                int tile_h2,
                int tile_w2,
                int pass_stride,  // bool use_load_balancing, int tile_step, int compress,
                int has_shadow_catcher,
                int max_shaders,
                unsigned int kernel_features,
                unsigned int volume_stack_size,
                DEVICE_PTR buffer_host_ptr,
                DEVICE_PTR pixels_host_ptr)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_path_trace);

  data.client_path_trace_data.buffer = (buffer_host_ptr != 0) ? buffer_host_ptr :
                                                                (DEVICE_PTR)buffer;
  data.client_path_trace_data.pixels = (pixels_host_ptr != 0) ? pixels_host_ptr :
                                                                (DEVICE_PTR)pixels;

  data.client_path_trace_data.start_sample = start_sample;
  data.client_path_trace_data.num_samples = num_samples;
  data.client_path_trace_data.sample_offset = sample_offset;

  data.client_path_trace_data.tile_x = tile_x;
  data.client_path_trace_data.tile_y = tile_y;
  data.client_path_trace_data.offset = offset;
  data.client_path_trace_data.stride = stride;
  data.client_path_trace_data.pass_stride = pass_stride;
  data.client_path_trace_data.tile_h = tile_h;
  data.client_path_trace_data.tile_w = tile_w;

  data.client_path_trace_data.has_shadow_catcher = has_shadow_catcher;
  data.client_path_trace_data.max_shaders = max_shaders;
  data.client_path_trace_data.kernel_features = kernel_features;
  data.client_path_trace_data.volume_stack_size = volume_stack_size;

  send_to_cache(data);
}

void receive_path_trace(int offset,
                        int stride,
                        int tile_x,
                        int tile_y,
                        int tile_h,
                        int tile_w,
                        int num_samples,
                        size_t pass_stride_sizeof,  // bool compress,
                        char *buffer_pixels,
                        char *task_bin,
                        char *tile_bin,
                        char *kg_bin,
                        void (*tex_update)(bool, char *, int, float, float, float, int, float *),
                        void (*update_progress)(char *, char *, int, int))
{
}

void receive_render_buffer(char *buffer_pixels, int tile_h, int tile_w, int pass_stride)
{
}

void send_data_cam(char *data, CLIENT_SIZE_T size, bool ack){};
void recv_data_cam(char *data, CLIENT_SIZE_T size, bool ack){};

void send_data_data(char *data, CLIENT_SIZE_T size, bool ack){};
void recv_data_data(char *data, CLIENT_SIZE_T size, bool ack){};

void client_close(){};
void server_close(){};
/////////////////////////////////////////////////////////////////////

// CCL_NAMESPACE_END
}  // namespace file
}  // namespace kernel
}  // namespace cyclesphi
