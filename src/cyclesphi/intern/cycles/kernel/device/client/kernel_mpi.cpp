#include "kernel_mpi.h"
#include "kernel_file.h"
#include "kernel_tcp.h"

#include <algorithm>
#include <map>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <vector>

#ifdef WITH_TURBOJPEG
#  include "turbojpeg.h"
#endif

#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
#  include <mpi.h>
#endif

//#include "util_texture.h"
//#include "util_types.h"
#define UCHAR4 uchar4

//#ifdef WITH_CLIENT_MPI_SOCKET
//
//#include <string.h>
//
//#ifdef WIN32
//#include <winsock2.h>
//#include <ws2tcpip.h>
//
//#pragma comment(lib, "Ws2_32.lib")
//#else
//#include <sys/socket.h>
//#include <netinet/in.h>
//#include <arpa/inet.h>
//#include <unistd.h>
//#include <netdb.h>
//#endif
//
//#include <iostream>
//#include <sys/types.h>
//#include <string>
//#include <cstdlib>
//
//#endif

// CCL_NAMESPACE_BEGIN

namespace cyclesphi {
namespace kernel {
namespace mpi {

#ifdef WITH_TURBOJPEG

tjhandle g_jpegCompressor = NULL;
tjhandle g_jpegDecompressor = NULL;

void turbojpeg_compression(int width,
                           int height,
                           unsigned char *image,
                           unsigned char **compressedImage,
                           uint64_t &jpegSize,
                           int JPEG_QUALITY,
                           int COLOR_COMPONENTS)
{
  // const int JPEG_QUALITY = 75;
  // const int COLOR_COMPONENTS = 4;
  // long unsigned int _jpegSize = 0;
  // unsigned char* _compressedImage = NULL; //!< Memory is allocated by tjCompress2 if _jpegSize
  // == 0 unsigned char buffer[_width*_height*COLOR_COMPONENTS]; //!< Contains the uncompressed
  // image

  if (g_jpegCompressor == NULL)
    g_jpegCompressor = tjInitCompress();

  if (g_jpegCompressor == NULL)
    printf("tjInitCompress: %s\n", tjGetErrorStr());

  unsigned long s = 0;
  int res = tjCompress2(g_jpegCompressor,
                        image,
                        width,
                        0,
                        height,
                        TJPF_RGBA,
                        compressedImage,
                        &s,
                        TJSAMP_444,
                        JPEG_QUALITY,
                        TJFLAG_FASTDCT);

  if (res == -1)
    printf("tjCompress2: %s\n", tjGetErrorStr());

  jpegSize = s;
  // tjDestroy(g_jpegCompressor);

  // to free the memory allocated by TurboJPEG (either by tjAlloc(),
  // or by the Compress/Decompress) after you are done working on it:
  // tjFree(_compressedImage);
}

void turbojpeg_free(unsigned char *image)
{
  tjFree(image);
}

void turbojpeg_decompression(int width,
                             int height,
                             uint64_t jpegSize,
                             unsigned char *compressedImage,
                             unsigned char *image,
                             int COLOR_COMPONENTS)
{
  // const int COLOR_COMPONENTS = 4;

  //    int jpegSubsamp, _width, _height;
  //[width*height*COLOR_COMPONENTS]; //!< will contain the decompressed image

  if (g_jpegDecompressor == NULL)
    g_jpegDecompressor = tjInitDecompress();

  if (g_jpegDecompressor == NULL)
    printf("tjInitDecompress: %s\n", tjGetErrorStr());
  //
  //    tjDecompressHeader2(g_jpegDecompressor, compressedImage, jpegSize, &_width, &_height,
  //    &jpegSubsamp);
  //	printf("g_jpegDecompressor: %ld, %d, %d, %d\n", (size_t)g_jpegDecompressor, jpegSubsamp,
  //_width, _height);

  // tjDecompress2(g_jpegDecompressor, compressedImage, (unsigned long) jpegSize, image, width,
  // 0/*pitch*/, height, TJPF_RGBA, TJFLAG_FASTDCT);
  int res = tjDecompress2(g_jpegDecompressor,
                          compressedImage,
                          (unsigned long)jpegSize,
                          image,
                          width,
                          0 /*pitch*/,
                          height,
                          TJPF_RGBA,
                          TJFLAG_FASTDCT);

  if (res == -1)
    printf("tjDecompress2: %s\n", tjGetErrorStr());

  // tjDestroy(g_jpegDecompressor);
}

#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MPI_Request *g_req = NULL;

#ifdef WITH_CLIENT_MPI_VRCLIENT
cyclesphi_data g_cdata;
#endif

#ifdef WITH_CLIENT_MPI_CACHE
#  define CACHING_DISABLED 0
#  define CACHING_RECORD 1
#  define CACHING_PREVIEW 2

int g_current_frame = 0;
int g_current_frame_preview = -1;
int g_caching_enabled = 0;
#endif

void socket_bcast(void *data, size_t size)
{
  // printf("BLENDER: bcast: %zu\n", size);

#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  const size_t unit_giga = 1024L * 1024L * 128L;

  size_t size_sended = 0;
  while (size - size_sended > unit_giga) {
    MPI_Bcast((char *)data + size_sended, unit_giga, MPI_BYTE, 0, MPI_COMM_WORLD);
    size_sended += unit_giga;
  }

  MPI_Bcast((char *)data + size_sended, size - size_sended, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)
  tcp::send_data_cam((char *)data, size);
#endif
}

int count_devices()
{
#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  return (world_size - 1 > 0) ? world_size - 1 : 1;
#endif

#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)
  return 1;
#endif
}

void get_kernel_data(client_kernel_struct *data, int tag)
{
  memset(data, 0, sizeof(client_kernel_struct));
  data->client_tag = tag;
}

// void send_kernel_data(client_kernel_struct *data) {
//    socket_bcast(data, sizeof (client_kernel_struct));
//}

////////////////////////////////////////////////////////////////////////////

struct MpiCache {
  client_kernel_struct data;
  char *mem;
  size_t size;
  bool valid;
  MpiCache(client_kernel_struct _data, char *_mem = NULL, size_t _size = 0)
      : data(_data), size(_size)
  {
    mem = NULL;
    if (_mem != NULL) {
      mem = new char[_size];
      memcpy(mem, _mem, _size);
    }
    valid = true;
  }
  ~MpiCache()
  {
    if (mem != NULL) {
      delete[] mem;
    }
  }
};

struct TexCache {
  char *info;
  char *mem;
  size_t size;
  bool valid;
  TexCache(char *_info, char *_mem = NULL, size_t _size = 0) : info(_info), size(_size)
  {
    mem = _mem;
    //        if(_mem!=NULL) {
    //            mem = new char[_size];
    //            memcpy(mem, _mem, _size);
    //        }
    valid = true;
  }
  ~TexCache()
  {
    //        if(mem!=NULL) {
    //            delete [] mem;
    //        }
  }
};

bool tex_sort(TexCache *i, TexCache *j)
{
  return (i->size < j->size);
}

std::vector<MpiCache *> g_caches;
std::vector<TexCache *> g_tex_caches;
int g_action = 0;
size_t g_mem_sum = 0;
size_t g_tex_mem_sum = 0;

void send_to_cache(client_kernel_struct &data, void *mem, size_t size)
{
#ifdef MPI_CACHE_FILE2
  // printf("send: %d\n", data.client_tag);

  switch (data.client_tag) {
    case CLIENT_TAG_CYCLES_alloc_kg: {
      g_action = 1;
      g_mem_sum = 0;
      g_tex_mem_sum = 0;
#  ifdef MPI_CACHE_FILE3
      file_write_data_kernelglobal(&data, sizeof(client_kernel_struct));
      if (mem != NULL)
        file_write_data_kernelglobal(mem, size);
#  else
      socket_bcast(&data, sizeof(client_kernel_struct));
      if (mem != NULL)
        socket_bcast(mem, size);
#  endif
    } break;
    case CLIENT_TAG_CYCLES_free_kg: {
      for (int i = 0; i < g_caches.size(); i++) {
        delete g_caches[i];
      }
      g_caches.clear();

      for (int i = 0; i < g_tex_caches.size(); i++) {
        delete g_tex_caches[i];
      }
      g_tex_caches.clear();

      g_action = 0;
      g_mem_sum = 0;
      g_tex_mem_sum = 0;
#  ifdef MPI_CACHE_FILE3
      file_write_data_kernelglobal(&data, sizeof(client_kernel_struct));
      if (mem != NULL)
        file_write_data_kernelglobal(mem, size);
      file_close_kernelglobal();
#  else
      socket_bcast(&data, sizeof(client_kernel_struct));
      if (mem != NULL)
        socket_bcast(mem, size);
#  endif
    } break;
    case CLIENT_TAG_CYCLES_path_trace: {
      g_action = 2;

      printf("blender: mem %zu, tex %zu\n", g_mem_sum, g_tex_mem_sum);

      double t = omp_get_wtime();

      std::sort(g_tex_caches.begin(), g_tex_caches.end(), tex_sort);

      size_t tex_size = 0;
      size_t tex_limit = 14L * 1024L * 1024L * 1024L;

      const char *sTexLimit = std::getenv("CLIENT_TEX_LIMIT");
      if (sTexLimit != NULL) {
        tex_limit = atoll(sTexLimit);
      }

      std::map<device_ptr, int> tex_map;
      for (int i = 0; i < g_tex_caches.size(); i++) {
        if (g_tex_caches[i]->valid == false)
          continue;
        printf("blender: tex check: %zu < %zu\n",
               g_tex_caches[i]->size + tex_size + g_mem_sum,
               tex_limit);
        if (g_tex_caches[i]->size + tex_size + g_mem_sum < tex_limit) {
          //                    printf("blender: local tex: %zu, %d, %d, %d\n",
          //                             g_tex_caches[i]->size,
          //                             ((TextureInfo*)g_tex_caches[i]->info)->width,
          //                             ((TextureInfo*)g_tex_caches[i]->info)->height,
          //                             ((TextureInfo*)g_tex_caches[i]->info)->depth);

          tex_size += g_tex_caches[i]->size;
          //((TextureInfo*)g_tex_caches[i]->info)->node_id = -1;

          // client_kernel_struct *client_data_a = new client_kernel_struct();
          mem_alloc((device_ptr)g_tex_caches[i]->mem,
                    g_tex_caches[i]->size,
                    NULL,
                    NULL /*client_data_a*/);

          // client_kernel_struct *client_data_c = new client_kernel_struct();
          mem_copy_to((device_ptr)g_tex_caches[i]->mem,
                      g_tex_caches[i]->size,
                      0,
                      NULL,
                      NULL /*client_data_c*/);
        }
        else {
          //((TextureInfo*)g_tex_caches[i]->info)->node_id = 0;
          tex_map[(device_ptr)g_tex_caches[i]->mem] = 0;
          printf("blender: remote tex: %zu, %d, %d, %d\n",
                 g_tex_caches[i]->size,
                 ((TextureInfo *)g_tex_caches[i]->info)->width,
                 ((TextureInfo *)g_tex_caches[i]->info)->height,
                 ((TextureInfo *)g_tex_caches[i]->info)->depth);
        }
      }

      for (int i = 0; i < g_caches.size(); i++) {
        if (g_caches[i]->valid == false)
          continue;

        if (g_caches[i]->data.client_tag == CLIENT_TAG_CYCLES_load_textures)
          printf("blender load tex size: %zu\n",
                 g_caches[i]->data.client_load_textures_data.texture_info_size);
        if (g_caches[i]->data.client_tag == CLIENT_TAG_CYCLES_tex_copy) {
          if (!strcmp(g_caches[i]->data.client_tex_copy_data.name, "texture_info")) {
            printf("blender client_tex_copy_data.data_size: %zu\n",
                   g_caches[i]->data.client_tex_copy_data.data_size);
            for (int j = 0; j < g_caches[i]->data.client_tex_copy_data.data_size; j++) {
              TextureInfo &info = ((TextureInfo *)g_caches[i]->mem)[j];
              info.node_id = -1;

              if (tex_map.find((device_ptr)info.data) != tex_map.end()) {
                info.node_id = tex_map[(device_ptr)info.data];
              }
            }
          }
        }

        // printf("Blender bcast: %d, %zu\n", g_caches[i]->data.client_tag,
        // g_caches[i]->size);

#  ifdef MPI_CACHE_FILE3
        file_write_data_kernelglobal(&(g_caches[i]->data), sizeof(client_kernel_struct));
        if (g_caches[i]->mem != NULL)
          file_write_data_kernelglobal(g_caches[i]->mem, g_caches[i]->size);
#  else
        socket_bcast(&(g_caches[i]->data), sizeof(client_kernel_struct));
        if (g_caches[i]->mem != NULL)
          socket_bcast(g_caches[i]->mem, g_caches[i]->size);
#  endif
      }

      printf("Blender bcast time: %f\n", omp_get_wtime() - t);
      fflush(0);

#  ifdef MPI_CACHE_FILE3
      file_close_kernelglobal();
#  endif
      socket_bcast(&data, sizeof(client_kernel_struct));
      if (mem != NULL)
        socket_bcast(mem, size);
    } break;

    case CLIENT_TAG_CYCLES_tex_copy:
    case CLIENT_TAG_CYCLES_mem_alloc:
      if (g_action == 1) {
        g_mem_sum += size;
      }

    case CLIENT_TAG_CYCLES_const_copy:
    case CLIENT_TAG_CYCLES_mem_copy_to:
    case CLIENT_TAG_CYCLES_mem_zero:
    case CLIENT_TAG_CYCLES_load_textures:
    case CLIENT_TAG_CYCLES_mem_alloc_sub_ptr: {
      if (g_action == 1) {
        g_caches.push_back(new MpiCache(data, (char *)mem, size));
      }
      else if (g_action != 2) {
        socket_bcast(&data, sizeof(client_kernel_struct));
        if (mem != NULL)
          socket_bcast(mem, size);
      }
    }

    // printf("Blender client mem-sum: %zu\n", g_mem_sum);
    break;

    case CLIENT_TAG_CYCLES_mem_free: {
      if (g_action == 1) {
        g_mem_sum -= size;
        for (int i = 0; i < g_caches.size(); i++) {
          if (data.client_mem_data.mem != 0 &&
              g_caches[i]->data.client_mem_data.mem == data.client_mem_data.mem)
            g_caches[i]->valid = false;
        }
      }
      else {
        socket_bcast(&data, sizeof(client_kernel_struct));
        if (mem != NULL)
          socket_bcast(mem, size);
      }
    } break;
    case CLIENT_TAG_CYCLES_tex_free: {
      if (g_action == 1) {
        g_mem_sum -= size;
        for (int i = 0; i < g_caches.size(); i++) {
          if (data.client_tex_copy_data.mem != 0 &&
              g_caches[i]->data.client_tex_copy_data.mem == data.client_tex_copy_data.mem)
            g_caches[i]->valid = false;
        }
      }
      else {
        socket_bcast(&data, sizeof(client_kernel_struct));
        if (mem != NULL)
          socket_bcast(mem, size);
      }
    } break;

    default:
      printf("Blender Cache: bad type: %d\n", data.client_tag);
      break;
  }
    // printf("send: %d\n", data.client_tag);
    // double t = omp_get_wtime();
#else
  socket_bcast(&data, sizeof(client_kernel_struct));

  if (mem != NULL)
    socket_bcast(mem, size);
#endif
  //    printf("BLENDER send: time: %f, tag: %d, size: %zu, memSize: %zu, tex_name: %s, tex_size:
  //    %zu\n", omp_get_wtime() - t,
  //            data.client_tag, size + sizeof(client_kernel_struct), data.client_mem_data.memSize,
  //            data.client_tex_copy_data.name, data.client_tex_copy_data.mem_size);
}

void path_to_cache(const char *path)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_path_to_cache);

  strcpy(data.client_const_copy_data.name, path);
  send_to_cache(data);
}

/////////////////////////////////////////////

//#ifdef CLIENT_USE_LOAD_BALANCING_SAMPLES
//
// char *g_kernel_pixels = NULL;
// size_t g_kernel_pixels_size = 0;
//
// char *g_kernel_pixels_tmp = NULL;
// size_t g_kernel_pixels_tmp_size = 0;
//
// int g_kernel_tile_h = 0;
// int g_kernel_tile_w = 0;
//
// void receive_path_trace_interactive2(int offset, int stride, int tile_x, int tile_y, int
// tile_h, int tile_w, int sample, int num_samples, size_t pass_stride_sizeof, bool compress, char*
// buffer_pixels, char* task_bin, char* tile_bin, void (*update_progress)(char*, char*, int, int))
// {
//#ifdef WITH_CLIENT_MPI
//
////    const int dev_count = count_devices();
////
////    int tile_step = tile_h / dev_count;
////    int tile_last = tile_h - (dev_count - 1) * tile_step;
////
////    const int dev_countAll = dev_count + 1;
////    std::vector<int> displsPix(dev_countAll);
////    std::vector<int> recvcountsPix(dev_countAll);
////    displsPix[0] = 0;
////    recvcountsPix[0] = 0;
////
////    for (int dev = 0; dev < dev_count; dev++) {
////        int tile_y2 = tile_y + tile_step * dev;
////        int tile_h2 = (dev_count - 1 == dev) ? tile_last : tile_step;
////
////        displsPix[dev + 1] = (offset + tile_x + tile_y2 * stride) * pass_stride_sizeof;
////        recvcountsPix[dev + 1] = tile_w * tile_h2 * pass_stride_sizeof;
////    }
//
//    ////////////////////////////one node///////////////////////////////////
//    //int dev_node = 0;
//    int devices_size_node = count_devices();
//
//    int tile_h_step_node = tile_h / devices_size_node;
//    int tile_h_last_node = tile_h - (devices_size_node - 1) * tile_h_step_node;
//
//    //int tile_y_node = data.client_path_trace_data.tile_y + tile_h_step_node * dev_node;
//    //int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_h_last_node :
//    tile_h_step_node;
//
//    //////////////////////////////////////////////////////
//    if (g_kernel_pixels != NULL && (g_kernel_tile_h != tile_h || g_kernel_tile_w !=
//    tile_w)) {
//        delete[] g_kernel_pixels;
//        delete[] g_kernel_pixels_tmp;
//
//        g_kernel_pixels = NULL;
//        g_kernel_tile_h = 0;
//        g_kernel_tile_w = 0;
//        g_kernel_pixels_size = 0;
//        g_kernel_pixels_tmp_size = 0;
//    }
//
//    if (g_kernel_pixels == NULL) {
//        g_kernel_pixels_size = (devices_size_node + 3) * tile_h_step_node * tile_w *
//        SIZE_UCHAR4; //(tile_h + tile_h_step_node /*rank0*/ + (tile_h_step_node -
//        tile_h_last_node) /*rankN*/) * tile_w * SIZE_UCHAR4; g_kernel_pixels_tmp_size =
//        tile_h_step_node * tile_w * SIZE_UCHAR4;
//
//        g_kernel_pixels = new char[g_kernel_pixels_size];
//        g_kernel_pixels_tmp = new char[g_kernel_pixels_tmp_size];
//
//        g_kernel_tile_h = tile_h;
//        g_kernel_tile_w = tile_w;
//    }
//    /////////////////////////////////////////////////////
//    //size_t offsetPix_node = (offset + tile_x + tile_y_node * stride) * SIZE_UCHAR4;
//    //size_t sizePix_node = tile_h_step_node * tile_w * SIZE_UCHAR4;
//
//    /////////////////////////////////////////////////////
//    //printf("%d: g_kernel_pixels_tmp_size: %zu, g_kernel_pixels_size: %zu, pixels_size:
//    %zu\n", 0, g_kernel_pixels_tmp_size, g_kernel_pixels_size, tile_w * tile_h *
//    pass_stride_sizeof);
//
//    MPI_Gather(g_kernel_pixels_tmp, g_kernel_pixels_tmp_size, MPI_BYTE,
//    g_kernel_pixels, g_kernel_pixels_tmp_size, MPI_BYTE, 0, MPI_COMM_WORLD);
//
//    memcpy(buffer_pixels, g_kernel_pixels + g_kernel_pixels_tmp_size, tile_w * tile_h *
//    pass_stride_sizeof);
//
//    (*update_progress)(task_bin, tile_bin, sample + num_samples, num_samples * tile_h * tile_w);
//
//#endif
//}
//
//
// void receive_path_trace_interactive(int offset, int stride, int tile_x, int tile_y, int
// tile_h, int tile_w, int sample, int num_samples, size_t pass_stride_sizeof, bool compress, char*
// buffer_pixels, char* task_bin, char* tile_bin, void (*update_progress)(char*, char*, int, int))
// {
//#ifdef WITH_CLIENT_MPI
//
//    const int dev_count = count_devices();
//
//    int tile_step = tile_h / dev_count;
//    int tile_last = tile_h - (dev_count - 1) * tile_step;
//
//    const int dev_countAll = dev_count + 1;
//    std::vector<int> displsPix(dev_countAll);
//    std::vector<int> recvcountsPix(dev_countAll);
//    displsPix[0] = 0;
//    recvcountsPix[0] = 0;
//
//    for (int dev = 0; dev < dev_count; dev++) {
//        int tile_y2 = tile_y + tile_step * dev;
//        int tile_h2 = (dev_count - 1 == dev) ? tile_last : tile_step;
//
//        displsPix[dev + 1] = (offset + tile_x + tile_y2 * stride) * pass_stride_sizeof;
//        recvcountsPix[dev + 1] = tile_w * tile_h2 * pass_stride_sizeof;
//    }
//
//    MPI_Gatherv(NULL, 0, MPI_BYTE, buffer_pixels, &recvcountsPix[0], &displsPix[0], MPI_BYTE, 0,
//    MPI_COMM_WORLD);
//
//    (*update_progress)(task_bin, tile_bin, sample + num_samples, num_samples * tile_h * tile_w);
//
//#endif
//}
//
//    void receive_path_trace_offline(int offset, int stride, int tile_x, int tile_y, int
//    tile_h,
//                                        int tile_w, int tile_h2, int tile_w2, int num_samples,
//                                        int tile_step, size_t pass_stride_sizeof, bool compress,
//                                        char* buffer_pixels, char* task_bin, char* task_pool_bin,
//                                        char* tile_bin, void(*update_progress)(char*, char*, int,
//                                        int), bool (*update_break)(char* task_bin, char*
//                                        task_pool_bin)) {
//
//#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
//#if 1
//        std::vector<char> buffer_temp(tile_w * tile_h * pass_stride_sizeof);
//        memset(&buffer_temp[0], 0, tile_w * tile_h * pass_stride_sizeof);
//
//        //memset((char*)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
//        MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//        //MPI_Allreduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
//
//        (*update_progress)(task_bin, tile_bin, num_samples, num_samples * tile_h * tile_w);
//#else
//        const int dev_count = count_devices();
//
//        const int dev_countAll = dev_count + 1;
//
//        std::vector<int> node_finished(dev_count);
//        std::vector<int> displsSample(dev_countAll);
//        std::vector<int> recvcountsSample(dev_countAll);
//        displsSample[0] = 0;
//        recvcountsSample[0] = 0;
//
//        std::vector<int> reqJob(dev_count);
//        std::vector<int> displsJob(dev_countAll);
//        std::vector<int> sendcountsJob(dev_countAll);
//        displsJob[0] = 0;
//        sendcountsJob[0] = 0;
//
//        int tile_i_node = dev_count;
//        int tile_i_count = num_samples;
//
//        for (int dev = 0; dev < dev_count; dev++) {
//            displsSample[dev + 1] = dev * sizeof (int);
//            recvcountsSample[dev + 1] = sizeof (int);
//
//            displsJob[dev + 1] = dev * sizeof (int);
//            sendcountsJob[dev + 1] = sizeof (int);
//            reqJob[dev] = -1;
//        }
//
//        int reqFinished = CLIENT_REQUEST_NONE;
//        double start_time = omp_get_wtime();
//
//        std::vector<char> buffer_temp(tile_w * tile_h * pass_stride_sizeof);
//        memset(&buffer_temp[0], 0, tile_w * tile_h * pass_stride_sizeof);
//
//        double timeReqImage = omp_get_wtime();
//
//        int count = 0;
//        while (true) {
//            MPI_Gatherv(NULL, 0, MPI_BYTE, &node_finished[0], &recvcountsSample[0],
//            &displsSample[0], MPI_BYTE, 0, MPI_COMM_WORLD);
//
//
//
//            MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//            pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//            (*update_progress)(task_bin, tile_bin, tile_i_node, 1 * tile_h * tile_w);
//
//
//            count++;
//
//
//            if (reqFinished == CLIENT_REQUEST_FINISH) {
//                for (int i = 0; i < dev_count; i++) {
//                    reqJob[i] = -2;
//                }
//            }
//
//            if (omp_get_wtime() - timeReqImage > 2.0 && reqFinished != CLIENT_REQUEST_FINISH) {
//                //  reqFinished = CLIENT_REQUEST_IMAGE;
//            }
//
//            MPI_Scatterv(&reqJob[0], &sendcountsJob[0], &displsJob[0], MPI_BYTE, NULL, 0,
//            MPI_BYTE, 0, MPI_COMM_WORLD); MPI_Bcast(&reqFinished, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//            if (reqFinished == CLIENT_REQUEST_FINISH) {
//                break;
//            }
//
//            if (min_count == 0 && tile_i_node > tile_i_count || update_break(task_bin,
//            task_pool_bin)) {
//                reqFinished = CLIENT_REQUEST_FINISH;
//            }
//        }
//
//        //memset((char*)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
//        //MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//        MPI_Allreduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
//
//        (*update_progress)(task_bin, tile_bin, num_samples, num_samples * tile_h * tile_w);
//#endif
//#endif
//
//#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)
//
//        while (true) {
//        //socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
//        pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);
//
//#if defined(WITH_TURBOJPEG)
//        if (compress) {
//            double t000 = omp_get_wtime();
//            unsigned char* image = (unsigned char*) buffer_pixels + (offset + tile_x + tile_y *
//            stride) * pass_stride_sizeof; int width = tile_w; int height = tile_h;
//
//            uint64_t jpeg_size = 0;
//            double t00 = omp_get_wtime();
//            socket_recv_data((char*) &jpeg_size, sizeof (uint64_t));
//            double t0 = omp_get_wtime();
//            unsigned char *compressed_image = new unsigned char[jpeg_size];
//            double t1 = omp_get_wtime();
//            socket_recv_data((char*) compressed_image, sizeof (unsigned char) * jpeg_size);
//            double t2 = omp_get_wtime();
//            turbojpeg_decompression(width, height, jpeg_size, compressed_image, image, 4);
//
//            double t3 = omp_get_wtime();
//            delete[] compressed_image;
//            double t4 = omp_get_wtime();
//
//            printf("size: %d, init: %.3f, wait: %.3f, alloc: %.3f, recv_data: %.3f,
//            decompression: %.3f, free: %.3f\n", jpeg_size, t00 - t000, t0 - t00, t1 - t0, t2 -
//            t1, t3 - t2, t4 - t3);
//        } else
//            socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
//            pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);
//#else
//        socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
//        pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);
//#endif
//
//        int pixel_sample_finished = 0;
//        socket_recv_data((char*) &pixel_sample_finished, sizeof (int));
//
//        if (pixel_sample_finished == -1)
//            break;
//
//        (*update_progress)(task_bin, tile_bin, num_samples, pixel_sample_finished);
//    }
//
//#endif
//    }
//
// void receive_path_trace_offline2(int offset, int stride, int tile_x, int tile_y, int tile_h,
//                                               int tile_w, int tile_h2, int tile_w2, int
//                                               num_samples, int tile_step, size_t
//                                               pass_stride_sizeof, bool compress, char*
//                                               buffer_pixels, char* task_bin, char*
//                                               task_pool_bin, char* tile_bin,
//                                               void(*update_progress)(char*, char*, int, int),
//                                               bool (*update_break)(char* task_bin, char*
//                                               task_pool_bin)) {
//
//#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
//        const int dev_count = count_devices();
//
//        const int dev_countAll = dev_count + 1;
//
//        std::vector<int> node_finished(dev_count);
//        std::vector<int> displsSample(dev_countAll);
//        std::vector<int> recvcountsSample(dev_countAll);
//        displsSample[0] = 0;
//        recvcountsSample[0] = 0;
//
//        std::vector<int> reqJob(dev_count);
//        std::vector<int> displsJob(dev_countAll);
//        std::vector<int> sendcountsJob(dev_countAll);
//        displsJob[0] = 0;
//        sendcountsJob[0] = 0;
//
//        int tile_i_node = dev_count;
//        int tile_i_count = num_samples;
//
//        for (int dev = 0; dev < dev_count; dev++) {
//            displsSample[dev + 1] = dev * sizeof (int);
//            recvcountsSample[dev + 1] = sizeof (int);
//
//            displsJob[dev + 1] = dev * sizeof (int);
//            sendcountsJob[dev + 1] = sizeof (int);
//            reqJob[dev] = -1;
//        }
//
//        int reqFinished = CLIENT_REQUEST_NONE;
//        double start_time = omp_get_wtime();
//
//        std::vector<char> buffer_temp(tile_w * tile_h * pass_stride_sizeof);
//        memset(&buffer_temp[0], 0, tile_w * tile_h * pass_stride_sizeof);
//
//        double timeReqImage = omp_get_wtime();
//
//        int count = 0;
//        while (true) {
//            MPI_Gatherv(NULL, 0, MPI_BYTE, &node_finished[0], &recvcountsSample[0],
//            &displsSample[0], MPI_BYTE, 0, MPI_COMM_WORLD);
//
//            //memset((char*)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
//            //MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//            pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//
//            if (reqFinished == CLIENT_REQUEST_IMAGE) {
////                MPI_Allreduce((char *) &buffer_temp[0], (char *) buffer_pixels,
////                              tile_w * tile_h * pass_stride_sizeof / sizeof(float), MPI_FLOAT,
/// MPI_SUM, MPI_COMM_WORLD);
//
//                MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//                pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//                (*update_progress)(task_bin, tile_bin, tile_i_node, 1 * tile_h * tile_w);
//                //(*update_progress)(NULL, NULL, 0, -1);
//
//                reqFinished = CLIENT_REQUEST_NONE;
//                timeReqImage = omp_get_wtime();
//            }
//
//            count++;
//            int min_count = 0;
//            for (int i = 0; i < dev_count; i++) {
//                if (node_finished[i] == 0 && tile_i_node <= tile_i_count) {
//                    reqJob[i] = tile_i_node;
//
//                    double current_time = omp_get_wtime();
//                    fprintf(stdout, "req_job: %d/%d, time: %.1f, remaining: %.1f, count:
//                    %d\n", tile_i_node, tile_i_count,
//                            current_time - start_time,
//                            (tile_i_count - tile_i_node) * (current_time - start_time) /
//                            (tile_i_count), count);
//
//                    fflush(stdout);
//
//                    tile_i_node++;
//                    count = 0;
//
////                    (*update_progress)(task_bin, tile_bin, tile_i_node, 1 * tile_h * tile_w);
////                    (*update_progress)(NULL, NULL, 0, 1 * tile_h * tile_w);
//
//                } else {
//                    reqJob[i] = -1;
//                }
//
//                if (min_count < node_finished[i])
//                    min_count = node_finished[i];
//            }
//
//            if (reqFinished == CLIENT_REQUEST_FINISH) {
//                for (int i = 0; i < dev_count; i++) {
//                    reqJob[i] = -2;
//                }
//            }
//
//            if (omp_get_wtime() - timeReqImage > 2.0 && reqFinished != CLIENT_REQUEST_FINISH) {
//              //  reqFinished = CLIENT_REQUEST_IMAGE;
//            }
//
//            MPI_Scatterv(&reqJob[0], &sendcountsJob[0], &displsJob[0], MPI_BYTE, NULL, 0,
//            MPI_BYTE, 0, MPI_COMM_WORLD); MPI_Bcast(&reqFinished, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//            if (reqFinished == CLIENT_REQUEST_FINISH) {
//                break;
//            }
//
//            if (min_count == 0 && tile_i_node > tile_i_count || update_break(task_bin,
//            task_pool_bin)) {
//                reqFinished = CLIENT_REQUEST_FINISH;
//            }
//        }
//
//        //memset((char*)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
//        MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//        //MPI_Allreduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
//        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
//
//        (*update_progress)(task_bin, tile_bin, num_samples, num_samples * tile_h * tile_w);
//        //(*update_progress)(NULL, NULL, 0, -1);
//
//        //    if (num_samples > 1)
//        //        printf("time: %f, remaining: %f, tile_y: %d/%d\n", omp_get_wtime() -
//        start_time, (tile_y + tile_h - tile_y_node) * (omp_get_wtime() - start_time) /
//        (tile_y_node - tile_y), tile_y_node, tile_y + tile_h);
//#endif
//
//#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)
//
//        while (true) {
//        //socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
//        pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);
//
//#if defined(WITH_TURBOJPEG)
//        if (compress) {
//            double t000 = omp_get_wtime();
//            unsigned char* image = (unsigned char*) buffer_pixels + (offset + tile_x + tile_y *
//            stride) * pass_stride_sizeof; int width = tile_w; int height = tile_h;
//
//            uint64_t jpeg_size = 0;
//            double t00 = omp_get_wtime();
//            socket_recv_data((char*) &jpeg_size, sizeof (uint64_t));
//            double t0 = omp_get_wtime();
//            unsigned char *compressed_image = new unsigned char[jpeg_size];
//            double t1 = omp_get_wtime();
//            socket_recv_data((char*) compressed_image, sizeof (unsigned char) * jpeg_size);
//            double t2 = omp_get_wtime();
//            turbojpeg_decompression(width, height, jpeg_size, compressed_image, image, 4);
//
//            double t3 = omp_get_wtime();
//            delete[] compressed_image;
//            double t4 = omp_get_wtime();
//
//            printf("size: %d, init: %.3f, wait: %.3f, alloc: %.3f, recv_data: %.3f,
//            decompression: %.3f, free: %.3f\n", jpeg_size, t00 - t000, t0 - t00, t1 - t0, t2 -
//            t1, t3 - t2, t4 - t3);
//        } else
//            socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
//            pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);
//#else
//        socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
//        pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);
//#endif
//
//        int pixel_sample_finished = 0;
//        socket_recv_data((char*) &pixel_sample_finished, sizeof (int));
//
//        if (pixel_sample_finished == -1)
//            break;
//
//        (*update_progress)(task_bin, tile_bin, num_samples, pixel_sample_finished);
//    }
//#endif
//
//    }
//
//
//    void path_trace(char *buffer, char *pixels,
//                        int start_sample, int num_samples, int tile_x,
//                        int tile_y, int offset, int stride, int tile_h, int tile_w, int tile_h2,
//                        int tile_w2, int pass_stride, bool use_load_balancing, int tile_step, int
//                        compress, DEVICE_PTR buffer_host_ptr, DEVICE_PTR pixels_host_ptr) {
//
//
//        client_kernel_struct data;
//        get_kernel_data(&data, CLIENT_TAG_CYCLES_path_trace);
//
//        data.client_path_trace_data.buffer = (buffer_host_ptr != NULL) ? buffer_host_ptr :
//        (DEVICE_PTR) buffer; data.client_path_trace_data.pixels = (pixels_host_ptr != NULL) ?
//        pixels_host_ptr : (DEVICE_PTR) pixels; data.client_path_trace_data.start_sample =
//        start_sample; data.client_path_trace_data.num_samples = num_samples;
//
//        data.client_path_trace_data.tile_x = tile_x;
//        data.client_path_trace_data.tile_y = tile_y;
//        data.client_path_trace_data.offset = offset;
//        data.client_path_trace_data.stride = stride;
//        data.client_path_trace_data.pass_stride = pass_stride;
//        data.client_path_trace_data.tile_h = tile_h;
//        data.client_path_trace_data.tile_w = tile_w;
//        data.client_path_trace_data.tile_h2 = tile_h2;
//        data.client_path_trace_data.tile_w2 = tile_w2;
//        data.client_path_trace_data.use_load_balancing = use_load_balancing;
//        data.client_path_trace_data.tile_step = tile_step;
//        data.client_path_trace_data.compress = compress;
//
//
//        send_to_cache(data);
//}
//#else

std::vector<char> receive_path_trace_buffer;

void receive_path_trace(int offset,
                        int stride,
                        int tile_x,
                        int tile_y,
                        int tile_h,
                        int tile_w,
                        int num_samples,
                        size_t pass_stride_sizeof,
                        // bool compress,
                        char *buffer_pixels,
                        char *task_bin,
                        char *tile_bin,
                        char *kg_bin,
                        void (*tex_update)(bool, char *, int, float, float, float, int, float *),
                        void (*update_progress)(char *, char *, int, int))
{
#ifdef WITH_CLIENT_MPI_VRCLIENT
  tile_w = stride = g_cdata.width;
  tile_h = g_cdata.height;
#endif

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  tile_w *= 2;  // left+right
  stride *= 2;  // left+right
#endif

#ifdef WITH_CLIENT_MPI_VRCLIENT
  if (receive_path_trace_buffer.size() != (tile_w * tile_h * pass_stride_sizeof)) {
    receive_path_trace_buffer.resize(tile_w * tile_h * pass_stride_sizeof);
  }
#endif

#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  const int dev_count = count_devices();

  // ceil
  int tile_step = (int)ceil((double)tile_h / (double)dev_count);
  int tile_last = tile_h - (dev_count - 1) * tile_step;
  if (tile_last < 1) {
    tile_step = (int)((double)tile_h / (double)(dev_count));
    tile_last = tile_h - (dev_count - 1) * tile_step;
  }
  const int dev_countAll = dev_count + 1;
  std::vector<int> displsPix(dev_countAll, 0);
  std::vector<int> recvcountsPix(dev_countAll, 0);
  // displsPix[0] = 0;
  // recvcountsPix[0] = 0;

  for (int dev = 0; dev < dev_count; dev++) {
    int tile_y2 = tile_y + tile_step * dev;
    int tile_h2 = (dev_count - 1 == dev) ? tile_last : tile_step;
    if (tile_h2 == 0)
      continue;
    displsPix[dev + 1] = (offset + tile_x + tile_y2 * stride) * pass_stride_sizeof;
    recvcountsPix[dev + 1] = tile_w * tile_h2 * pass_stride_sizeof;
  }

#  ifdef WITH_CLIENT_MPI_CACHE
  int f = (g_caching_enabled == CACHING_PREVIEW) ? g_current_frame_preview : g_current_frame;
  if (g_caching_enabled != CACHING_DISABLED) {
    for (int dev = 0; dev < dev_count; dev++) {
      if ((f % dev_count) == dev) {
        displsPix[dev + 1] = (offset + tile_x + tile_y * stride) * pass_stride_sizeof;
        recvcountsPix[dev + 1] = tile_w * tile_h * pass_stride_sizeof;
      }
      else {
        displsPix[dev + 1] = 0;
        recvcountsPix[dev + 1] = 0;
      }
    }
  }

#  endif

#  ifdef MPI_REMOTE_TEX
  client_tex_interp req;
  MPI_Status status;

  int finished_dev = 0;
  float resf4[4];
  while (finished_dev != dev_count) {
    // printf("blender MPI_Recv\n"); fflush(0);
    MPI_Recv(&req,
             sizeof(client_tex_interp),
             MPI_BYTE,
             MPI_ANY_SOURCE,
             MPI_ANY_TAG,
             MPI_COMM_WORLD,
             &status);

    // printf("blender received\n"); fflush(0);

    if (status.MPI_TAG == CLIENT_TAG_CYCLES_path_trace_finish) {
      finished_dev++;
    }

    if (status.MPI_TAG == CLIENT_TAG_CYCLES_tex_interp) {

      // printf("tex: %d, %d, %f, %f\n", status.MPI_SOURCE, req.id, (double)req.x, (double)req.y);
      (*tex_update)(false, kg_bin, req.id, req.x, req.y, req.z, req.type, resf4);
      // printf("blender MPI_Send\n"); fflush(0);

      MPI_Send(&resf4, 4, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);

      // printf("blender sended\n"); fflush(0);
    }

    if (status.MPI_TAG == CLIENT_TAG_CYCLES_tex_interp3d) {

      // printf("tex: %d, %d, %f, %f\n", status.MPI_SOURCE, req.id, (double)req.x, (double)req.y);
      (*tex_update)(true, kg_bin, req.id, req.x, req.y, req.z, req.type, resf4);
      // printf("blender MPI_Send\n"); fflush(0);

      MPI_Send(&resf4, 4, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);

      // printf("blender sended\n"); fflush(0);
    }
  }

  // printf("blender loop ended\n"); fflush(0);
  MPI_Barrier(MPI_COMM_WORLD);
#  endif

  //#ifdef CLIENT_MPI_LOAD_BALANCING_SAMPLES
  if (pass_stride_sizeof == sizeof(char) * 4 || pass_stride_sizeof == sizeof(short) * 4) {
    // MPI_Gatherv(NULL,
    //    0,
    //    MPI_BYTE,
    //    buffer_pixels,
    //    &recvcountsPix[0],
    //    &displsPix[0],
    //    MPI_BYTE,
    //    0,
    //    MPI_COMM_WORLD);
    MPI_Gatherv(NULL,
                0,
                MPI_BYTE,
#  ifdef WITH_CLIENT_MPI_VRCLIENT
                (char *)&receive_path_trace_buffer[0],
#  else
                buffer_pixels,
#  endif
                &recvcountsPix[0],
                &displsPix[0],
                MPI_BYTE,
                0,
                MPI_COMM_WORLD);

#  ifndef WITH_CLIENT_MPI_VRCLIENT
    int global_min;  // = pix_state[3];
    int local_min = 0;

    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    num_samples = global_min;
#  else
    socket_send_data_data((char *)&receive_path_trace_buffer[0], receive_path_trace_buffer.size());
    num_samples = 1;
#  endif
  }
  else {
    std::vector<char> buffer_temp(tile_w * tile_h * pass_stride_sizeof);
    memset(&buffer_temp[0], 0, tile_w * tile_h * pass_stride_sizeof);
    memset((char *)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
    MPI_Reduce((char *)&buffer_temp[0],
               (char *)buffer_pixels,
               tile_w * tile_h * pass_stride_sizeof / sizeof(float),
               MPI_FLOAT,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);
  }
//#else
//  MPI_Gatherv(NULL,
//              0,
//              MPI_BYTE,
//#ifdef WITH_CLIENT_MPI_VRCLIENT
//              (char *)&receive_path_trace_buffer[0],
//#else
//              buffer_pixels,
//#endif
//              &recvcountsPix[0],
//              &displsPix[0],
//              MPI_BYTE,
//              0,
//              MPI_COMM_WORLD);
//
//#    ifndef WITH_CLIENT_MPI_VRCLIENT
//  int global_min;// = pix_state[3];
//  int local_min = 0;
//
//  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0,
//                       MPI_COMM_WORLD);
//  num_samples = global_min;
//#else
//  socket_send_data_data((char *)&receive_path_trace_buffer[0],
//                   receive_path_trace_buffer.size());
//  num_samples = 1;
//#    endif
//
//#endif
#endif

#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT) && \
    !defined(WITH_CLIENT_RENDERENGINE_SENDER)

#  if defined(WITH_TURBOJPEG)
  if (compress) {
    double t000 = omp_get_wtime();
    unsigned char *image = (unsigned char *)buffer_pixels +
                           (offset + tile_x + tile_y * stride) * pass_stride_sizeof;
    int width = tile_w;
    int height = tile_h;

    uint64_t jpeg_size = 0;
    double t00 = omp_get_wtime();
    socket_recv_data((char *)&jpeg_size, sizeof(uint64_t));
    double t0 = omp_get_wtime();
    unsigned char *compressed_image = new unsigned char[jpeg_size];
    double t1 = omp_get_wtime();
    socket_recv_data((char *)compressed_image, sizeof(unsigned char) * jpeg_size);
    double t2 = omp_get_wtime();
    turbojpeg_decompression(width, height, jpeg_size, compressed_image, image, 4);

    // FILE *f = fopen("c:/temp/test.jpg", "wb");
    // fwrite((char*)compressed_image, sizeof (unsigned char) * jpeg_size, 1, f);
    // fclose(f);

    double t3 = omp_get_wtime();
    delete[] compressed_image;
    double t4 = omp_get_wtime();

    // printf("size: %d, init: %.3f, wait: %.3f, alloc: %.3f, recv_data: %.3f, decompression: %.3f,
    // free: %.3f\n", jpeg_size, t00 - t000, t0 - t00, t1 - t0, t2 - t1, t3 - t2, t4 - t3);
  }
  else
    socket_recv_data((char *)buffer_pixels +
                         (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                     tile_w * tile_h * pass_stride_sizeof);
#  else

  size_t u_char_4 = sizeof(unsigned char) * 4;
  size_t u_half_4 = sizeof(unsigned short) * 4;

  if (pass_stride_sizeof == u_char_4) {
#    ifdef WITH_CLIENT_GPUJPEG
    if (receive_path_trace_buffer.size() != (tile_w * tile_h * sizeof(unsigned char) * 4)) {
      receive_path_trace_buffer.resize(tile_w * tile_h * sizeof(unsigned char) * 4);
    }

    tcp::recv_gpujpeg((char *)buffer_pixels +
                          (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                      (char *)&receive_path_trace_buffer[0],
                      tile_w,
                      tile_h);
#    else
    socket_recv_data_data((char *)buffer_pixels +
                              (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                          tile_w * tile_h * pass_stride_sizeof);
#    endif
  }
  else if (pass_stride_sizeof == u_half_4) {
#    ifdef WITH_CLIENT_GPUJPEG
    if (receive_path_trace_buffer.size() != (tile_w * tile_h * sizeof(unsigned char) * 4)) {
      receive_path_trace_buffer.resize(tile_w * tile_h * sizeof(unsigned char) * 4);
    }

    tcp::recv_gpujpeg((char *)buffer_pixels +
                          (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                      (char *)&receive_path_trace_buffer[0],
                      tile_w,
                      tile_h);
#    else
    if (receive_path_trace_buffer.size() != (tile_w * tile_h * sizeof(unsigned char) * 4)) {
      receive_path_trace_buffer.resize(tile_w * tile_h * sizeof(unsigned char) * 4);
    }

    socket_recv_data_data((char *)&receive_path_trace_buffer[0], receive_path_trace_buffer.size());

    socket_rgb_to_half(
        (unsigned short *)((char *)buffer_pixels +
                           (offset + tile_x + tile_y * stride) * pass_stride_sizeof),
        (unsigned char *)&receive_path_trace_buffer[0],
        tile_h,
        tile_w);
#    endif
  }
  else {
    tcp::recv_data_data((char *)buffer_pixels +
                            (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                        tile_w * tile_h * pass_stride_sizeof);
  }

#  endif

    // socket_recv_data_data((char *)&num_samples, sizeof(int));
#  ifdef WITH_CLIENT_GPUJPEG
  num_samples = 1;
#  else
  num_samples = ((int *)buffer_pixels)[0];
#  endif

#endif

  // printf("num_samples: %d\n", num_samples);
  (*update_progress)(task_bin, tile_bin, num_samples, num_samples /** tile_h * tile_w*/);
}

void receive_path_trace_load_balancing(int offset,
                                       int stride,
                                       int tile_x,
                                       int tile_y,
                                       int tile_h,
                                       int tile_w,
                                       int tile_h2,
                                       int tile_w2,
                                       int num_samples,
                                       int tile_step,
                                       size_t pass_stride_sizeof,
                                       // bool compress,
                                       char *buffer_pixels,
                                       char *task_bin,
                                       char *task_pool_bin,
                                       char *tile_bin,
                                       void (*update_progress)(char *, char *, int, int),
                                       bool (*update_break)(char *task_bin, char *task_pool_bin))
{

#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  const int dev_count = count_devices();

  const int dev_countAll = dev_count + 1;
  //    std::vector<int> displsBuf(dev_countAll);
  //    std::vector<int> recvcountsBuf(dev_countAll);
  //    displsBuf[0] = 0;
  //    recvcountsBuf[0] = 0;

  std::vector<int> node_finished(dev_count);
  std::vector<int> displsSample(dev_countAll);
  std::vector<int> recvcountsSample(dev_countAll);
  displsSample[0] = 0;
  recvcountsSample[0] = 0;

  //    std::vector<int> row_finished(dev_count);
  //    std::vector<int> displsRow(dev_countAll);
  //    std::vector<int> recvcountsRow(dev_countAll);
  //    displsRow[0] = 0;
  //    recvcountsRow[0] = 0;

  std::vector<int> reqJob(dev_count);
  std::vector<int> displsJob(dev_countAll);
  std::vector<int> sendcountsJob(dev_countAll);
  displsJob[0] = 0;
  sendcountsJob[0] = 0;

  int tile_i_node = dev_count;
  int tile_x_count = (int)ceil((double)tile_w / (double)tile_w2);
  int tile_y_count = (int)ceil((double)tile_h / (double)tile_h2);
  int tile_i_count = tile_x_count * tile_y_count;

  for (int dev = 0; dev < dev_count; dev++) {
    displsSample[dev + 1] = dev * sizeof(int);
    recvcountsSample[dev + 1] = sizeof(int);

    //        displsRow[dev + 1] = dev * sizeof (int);
    //        recvcountsRow[dev + 1] = sizeof (int);

    displsJob[dev + 1] = dev * sizeof(int);
    sendcountsJob[dev + 1] = sizeof(int);
    reqJob[dev] = -1;
  }

  int reqFinished = 0;
  double start_time = 0;  // omp_get_wtime();

  std::vector<char> buffer_temp(tile_w * tile_h * pass_stride_sizeof);
  memset(&buffer_temp[0], 0, tile_w * tile_h * pass_stride_sizeof);

  int count = 0;
  while (true) {
    MPI_Gatherv(NULL,
                0,
                MPI_BYTE,
                &node_finished[0],
                &recvcountsSample[0],
                &displsSample[0],
                MPI_BYTE,
                0,
                MPI_COMM_WORLD);

    //        MPI_Gatherv(NULL, 0, MPI_BYTE, &row_finished[0], &recvcountsRow[0], &displsRow[0],
    //        MPI_BYTE, 0, MPI_COMM_WORLD);
    //
    //        for (int i = 0; i < dev_count; i++) {
    //            int tile_h2 = tile_step;
    //            if (tile_h2 + row_finished[i] > tile_h + tile_y)
    //                tile_h2 = tile_h + tile_y - row_finished[i];
    //
    ////            recvcountsBuf[i + 1] = tile_w * tile_h2 * pass_stride_sizeof;
    ////
    ////            displsBuf[i + 1] = (offset + tile_x + row_finished[i] * stride) *
    /// pass_stride_sizeof;
    //        }

    // MPI_Gatherv(NULL, 0, MPI_BYTE, (char*) buffer_pixels, &recvcountsBuf[0], &displsBuf[0],
    // MPI_BYTE, 0, MPI_COMM_WORLD);

    //        memset((char*)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
    //        MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h *
    //        pass_stride_sizeof / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    count++;
    int min_count = 0;
    for (int i = 0; i < dev_count; i++) {
      if (node_finished[i] == 0 && tile_i_node <= tile_i_count) {
        reqJob[i] = tile_i_node;
        // node_finished[i] = 1;

        double current_time = 0;  // omp_get_wtime();
        fprintf(stdout,
                "req_job: %d/%d, time: %.1f, remaining: %.1f, count: %d\n",
                tile_i_node,
                tile_i_count,
                current_time - start_time,
                (tile_i_count - tile_i_node) * (current_time - start_time) / (tile_i_count),
                count);

        fflush(stdout);

        tile_i_node++;
        count = 0;

        //                if (tile_y_node > tile_y + tile_h)
        //                    tile_y_node = tile_y + tile_h;

        // task.update_progress(&tile, tile.num_samples * 1 * tile.w);
        (*update_progress)(task_bin, tile_bin, num_samples, num_samples * tile_h2 * tile_w2);
        (*update_progress)(NULL, NULL, 0, num_samples * tile_h2 * tile_w2);

        //                double current_time = omp_get_wtime();
        //                if (tile_y_node > dev_count && num_samples > 1)
        //                    printf("time: %f, remaining: %f, tile_y: %d/%d\n", current_time -
        //                    start_time, (tile_y + tile_h - tile_y_node + dev_count) *
        //                    (current_time - start_time) / (tile_y_node - dev_count - tile_y),
        //                    tile_y_node - dev_count, tile_y + tile_h);
      }
      else {
        reqJob[i] = -1;
      }

      if (min_count < node_finished[i])
        min_count = node_finished[i];
    }

    if (reqFinished != 0) {
      for (int i = 0; i < dev_count; i++) {
        reqJob[i] = -2;
      }
    }

    MPI_Scatterv(&reqJob[0],
                 &sendcountsJob[0],
                 &displsJob[0],
                 MPI_BYTE,
                 NULL,
                 0,
                 MPI_BYTE,
                 0,
                 MPI_COMM_WORLD);
    MPI_Bcast(&reqFinished, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (reqFinished != 0) {
      break;
    }

    if (min_count == 0 && tile_i_node > tile_i_count || update_break(task_bin, task_pool_bin)) {
      reqFinished = 1;
      //                for (int i = tile_y; i < tile_y + tile_h; i++) {
      //                    char *b = (char*) buffer_pixels + (offset + tile_x + i * stride) *
      //                    pass_stride_sizeof; if (b[tile_w - 1] == char(-1) && b[tile_w - 2] ==
      //                    char(-1) && b[tile_w - 3] == char(-1)) {
      //                        reqFinished = 0;
      //                        break;
      //                    }
      //                }
    }

    //            if (task.get_cancel() || task_pool.canceled()) {
    //                if (task.need_finish_queue == false)
    //                    reqFinished = 1;
    //            }
  }

  memset((char *)buffer_pixels, 0, tile_w * tile_h * pass_stride_sizeof);
  MPI_Reduce((char *)&buffer_temp[0],
             (char *)buffer_pixels,
             tile_w * tile_h * pass_stride_sizeof / sizeof(float),
             MPI_FLOAT,
             MPI_SUM,
             0,
             MPI_COMM_WORLD);

  (*update_progress)(task_bin, tile_bin, num_samples, num_samples * tile_h * tile_w);
  (*update_progress)(NULL, NULL, 0, -1);

  //    if (num_samples > 1)
  //        printf("time: %f, remaining: %f, tile_y: %d/%d\n", omp_get_wtime() - start_time,
  //        (tile_y + tile_h - tile_y_node) * (omp_get_wtime() - start_time) / (tile_y_node -
  //        tile_y), tile_y_node, tile_y + tile_h);
#endif

#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)

  while (true) {
    // socket_recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
    // pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);

#  if defined(WITH_TURBOJPEG)
    if (compress) {
      double t000 = omp_get_wtime();
      unsigned char *image = (unsigned char *)buffer_pixels +
                             (offset + tile_x + tile_y * stride) * pass_stride_sizeof;
      int width = tile_w;
      int height = tile_h;

      uint64_t jpeg_size = 0;
      double t00 = omp_get_wtime();
      socket_recv_data((char *)&jpeg_size, sizeof(uint64_t));
      double t0 = omp_get_wtime();
      unsigned char *compressed_image = new unsigned char[jpeg_size];
      double t1 = omp_get_wtime();
      socket_recv_data((char *)compressed_image, sizeof(unsigned char) * jpeg_size);
      double t2 = omp_get_wtime();
      turbojpeg_decompression(width, height, jpeg_size, compressed_image, image, 4);

      double t3 = omp_get_wtime();
      delete[] compressed_image;
      double t4 = omp_get_wtime();

      printf(
          "size: %d, init: %.3f, wait: %.3f, alloc: %.3f, recv_data: %.3f, decompression: %.3f, "
          "free: %.3f\n",
          jpeg_size,
          t00 - t000,
          t0 - t00,
          t1 - t0,
          t2 - t1,
          t3 - t2,
          t4 - t3);
    }
    else
      socket_recv_data((char *)buffer_pixels +
                           (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                       tile_w * tile_h * pass_stride_sizeof);
#  else
    tcp::recv_data_data((char *)buffer_pixels +
                            (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                        tile_w * tile_h * pass_stride_sizeof);
#  endif

    int pixel_sample_finished = 0;
    tcp::recv_data_data((char *)&pixel_sample_finished, sizeof(int));

    if (pixel_sample_finished == -1)
      break;

    (*update_progress)(task_bin, tile_bin, num_samples, pixel_sample_finished);
  }

#endif
}

//////////////////////////////////////////////////////////////////////
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
                int pass_stride,
                // bool use_load_balancing, int tile_step, int compress,
                int has_shadow_catcher,
                int max_shaders,
                unsigned int kernel_features,
                unsigned int volume_stack_size,
                DEVICE_PTR buffer_host_ptr,
                DEVICE_PTR pixels_host_ptr)
{

  //    printf("samples: %d/%d\n", start_sample, num_samples);
  //    fflush(0);

  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_path_trace);

#ifdef WITH_CLIENT_MPI_VRCLIENT
  socket_recv_data_cam((char *)&g_cdata, sizeof(cyclesphi_data));
  memcpy(&data.client_path_trace_data.cdata, (char *)&g_cdata, sizeof(cyclesphi_data));
  tile_w2 = tile_w = stride = g_cdata.width;
  tile_h2 = tile_h = g_cdata.height;
  num_samples = g_cdata.step_samples;
#endif

  data.client_path_trace_data.buffer = (buffer_host_ptr != NULL) ? buffer_host_ptr :
                                                                   (DEVICE_PTR)buffer;
  data.client_path_trace_data.pixels = (pixels_host_ptr != NULL) ? pixels_host_ptr :
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
  data.client_path_trace_data.tile_h2 = tile_h2;
  data.client_path_trace_data.tile_w2 = tile_w2;
  // data.client_path_trace_data.use_load_balancing = use_load_balancing;
  // data.client_path_trace_data.tile_step = tile_step;
  // data.client_path_trace_data.compress = compress;
  data.client_path_trace_data.has_shadow_catcher = has_shadow_catcher;
  data.client_path_trace_data.max_shaders = max_shaders;
  data.client_path_trace_data.kernel_features = kernel_features;
  data.client_path_trace_data.volume_stack_size = volume_stack_size;

  send_to_cache(data);
}

//#endif

void receive_render_buffer(char *buffer_pixels, int tile_h, int tile_w, int pass_stride)
{

  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_receive_render_buffer);

  data.client_path_trace_data.buffer = (DEVICE_PTR)buffer_pixels;
  data.client_path_trace_data.pass_stride = pass_stride;
  data.client_path_trace_data.tile_h = tile_h;
  data.client_path_trace_data.tile_w = tile_w;

  send_to_cache(data);

  // std::vector<char> buffer_temp(tile_w * tile_h * pass_stride * sizeof(float));
  // memset(&buffer_temp[0], 0, tile_w * tile_h * pass_stride * sizeof(float));
  memset(buffer_pixels, 0, tile_w * tile_h * pass_stride * sizeof(float));

  // MPI_Reduce((char*) &buffer_temp[0], (char*) buffer_pixels, tile_w * tile_h * pass_stride,
  // MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  const int dev_count = count_devices();

  int tile_step = tile_w / dev_count;
  int tile_last = tile_w - (dev_count - 1) * tile_step;

  // int tile_x = 0;
  // int tile_y = 0;
  // int offset = 0;
  // int stride = tile_w;

  const int dev_countAll = dev_count + 1;
  std::vector<int> displsPix(dev_countAll);
  std::vector<int> recvcountsPix(dev_countAll);
  displsPix[0] = 0;
  recvcountsPix[0] = 0;

  for (int dev = 0; dev < dev_count; dev++) {
    // int tile_x2 = tile_x + tile_step * dev;
    int tile_w2 = (dev_count - 1 == dev) ? tile_last : tile_step;

    displsPix[dev + 1] = dev * tile_w2 * tile_h * pass_stride * sizeof(float);
    recvcountsPix[dev + 1] = tile_w2 * tile_h * pass_stride * sizeof(float);
  }

#if defined(WITH_CLIENT_NCCL_SOCKET)

#elif defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  MPI_Gatherv(NULL,
              0,
              MPI_BYTE,
              buffer_pixels,
              &recvcountsPix[0],
              &displsPix[0],
              MPI_BYTE,
              0,
              MPI_COMM_WORLD);
#endif
}

void const_copy(const char *name,
                char *host_bin,
                size_t size,
                DEVICE_PTR host_ptr,
                client_kernel_struct *_client_data)
{
  if (strcmp(name, "data") == 0) {
    client_kernel_struct data;
    get_kernel_data(&data, CLIENT_TAG_CYCLES_const_copy);

    strcpy(data.client_const_copy_data.name, name);
    data.client_const_copy_data.host = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)host_bin;
    data.client_const_copy_data.size = size;
    data.client_const_copy_data.read_data = true;
    if (_client_data != NULL)
      memcpy(_client_data, &data, sizeof(client_kernel_struct));
    else
      send_to_cache(data, host_bin, size);
  }
}

void load_textures(size_t size, client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_load_textures);

  data.client_load_textures_data.texture_info_size = size;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data);
}

void tex_copy(const char *name,
              void *mem,
              size_t data_size,
              size_t mem_size,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{

  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_tex_copy);

  strcpy(data.client_tex_copy_data.name, name);
  data.client_tex_copy_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  data.client_tex_copy_data.data_size = data_size;
  data.client_tex_copy_data.mem_size = mem_size;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data, (char *)mem, mem_size);
}

void alloc_kg(client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_alloc_kg);
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data);
}

void free_kg(client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_free_kg);
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
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
  data.client_mem_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data);
}

void mem_alloc_sub_ptr(const char *name,
                       DEVICE_PTR mem,
                       size_t offset,
                       DEVICE_PTR mem_sub,
                       client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_mem_alloc_sub_ptr);

  strcpy(data.client_mem_data.name, name);
  data.client_mem_data.mem = mem;
  data.client_mem_data.offset = offset;
  data.client_mem_data.mem_sub = mem_sub;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data);
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
  data.client_mem_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;
  data.client_mem_data.offset = offset;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
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
  data.client_mem_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;
  data.client_mem_data.offset = offset;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
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
  data.client_mem_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
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
  data.client_mem_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  data.client_mem_data.memSize = memSize;

  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data);
}
/////////////////////////////////////////////////////////////////////

// void denoising_non_local_means(DEVICE_PTR image_ptr, DEVICE_PTR guide_ptr, DEVICE_PTR
// variance_ptr, DEVICE_PTR out_ptr,
//                                   client_denoising_task_struct *task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_non_local_means);

//    data.client_denoising_non_local_means_data.image_ptr = image_ptr;
//    data.client_denoising_non_local_means_data.guide_ptr = guide_ptr;
//    data.client_denoising_non_local_means_data.variance_ptr = variance_ptr;
//    data.client_denoising_non_local_means_data.out_ptr = out_ptr;

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

// void denoising_construct_transform(client_denoising_task_struct *task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_construct_transform);

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

// void denoising_reconstruct(DEVICE_PTR color_ptr,
//                               DEVICE_PTR color_variance_ptr,
//                               DEVICE_PTR output_ptr,
//                               client_denoising_task_struct *task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_reconstruct);

//    data.client_denoising_reconstruct_data.color_ptr = color_ptr;
//    data.client_denoising_reconstruct_data.color_variance_ptr = color_variance_ptr;
//    data.client_denoising_reconstruct_data.output_ptr = output_ptr;

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

// void denoising_combine_halves(DEVICE_PTR a_ptr, DEVICE_PTR b_ptr,
//                                  DEVICE_PTR mean_ptr, DEVICE_PTR variance_ptr,
//                                  int r, int* rect, client_denoising_task_struct * task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_combine_halves);

//    data.client_denoising_combine_halves_data.a_ptr = a_ptr;
//    data.client_denoising_combine_halves_data.b_ptr = b_ptr;
//    data.client_denoising_combine_halves_data.mean_ptr = mean_ptr;
//    data.client_denoising_combine_halves_data.variance_ptr = variance_ptr;
//    data.client_denoising_combine_halves_data.r = r;
//    data.client_denoising_combine_halves_data.rect[0] = rect[0];
//    data.client_denoising_combine_halves_data.rect[1] = rect[1];
//    data.client_denoising_combine_halves_data.rect[2] = rect[2];
//    data.client_denoising_combine_halves_data.rect[3] = rect[3];

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

// void denoising_divide_shadow(DEVICE_PTR a_ptr, DEVICE_PTR b_ptr,
//                                 DEVICE_PTR sample_variance_ptr, DEVICE_PTR sv_variance_ptr,
//                                 DEVICE_PTR buffer_variance_ptr, client_denoising_task_struct
//                                 *task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_divide_shadow);

//    data.client_denoising_divide_shadow_data.a_ptr = a_ptr;
//    data.client_denoising_divide_shadow_data.b_ptr = b_ptr;
//    data.client_denoising_divide_shadow_data.sample_variance_ptr = sample_variance_ptr;
//    data.client_denoising_divide_shadow_data.sv_variance_ptr = sv_variance_ptr;
//    data.client_denoising_divide_shadow_data.buffer_variance_ptr = buffer_variance_ptr;

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

// void denoising_get_feature(int mean_offset,
//                               int variance_offset,
//                               DEVICE_PTR mean_ptr,
//                               DEVICE_PTR variance_ptr,
//                               client_denoising_task_struct *task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_get_feature);

//    data.client_denoising_get_feature_data.mean_offset = mean_offset;
//    data.client_denoising_get_feature_data.variance_offset = variance_offset;
//    data.client_denoising_get_feature_data.mean_ptr = mean_ptr;
//    data.client_denoising_get_feature_data.variance_ptr = variance_ptr;

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

// void denoising_detect_outliers(DEVICE_PTR image_ptr,
//                                   DEVICE_PTR variance_ptr,
//                                   DEVICE_PTR depth_ptr,
//                                   DEVICE_PTR output_ptr,
//                                   client_denoising_task_struct *task){
//    client_kernel_struct data;
//    get_kernel_data(&data, CLIENT_TAG_CYCLES_denoising_detect_outliers);

//    data.client_denoising_detect_outliers_data.image_ptr = image_ptr;
//    data.client_denoising_detect_outliers_data.variance_ptr = variance_ptr;
//    data.client_denoising_detect_outliers_data.depth_ptr = depth_ptr;
//    data.client_denoising_detect_outliers_data.output_ptr = output_ptr;

//    send_to_cache(data, task, sizeof(client_denoising_task_struct));
//}

/////////////////////////////////////////////////////////////////////
//
// struct lock_init_struct {
//    omp_lock_t writelock;
//    lock_init_struct() {printf("omp_init_lock,\n");omp_init_lock(&writelock);}
//    ~lock_init_struct() {printf("omp_destroy_lock,\n");omp_destroy_lock(&writelock);}
//};
// lock_init_struct lock_init_data;

//(&g_writelock);
void tex_image_interp(int id, float x, float y, float *result)
{
#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  client_tex_interp req;
  req.id = id;
  req.x = x;
  req.y = y;

#  pragma omp critical
  {
    // printf("client MPI_Send %d\n", omp_get_thread_num()); fflush(0);
    MPI_Send(&req,
             sizeof(client_tex_interp),
             MPI_BYTE,
             0,
             CLIENT_TAG_CYCLES_tex_interp,
             MPI_COMM_WORLD);
    // printf("client MPI_Recv %d\n", omp_get_thread_num()); fflush(0);
    MPI_Recv(
        result, 4, MPI_FLOAT, 0, CLIENT_TAG_CYCLES_tex_interp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // printf("client finish %d\n", omp_get_thread_num()); fflush(0);
  }
  // omp_unset_lock(&lock_init_data.writelock);
#endif
}

void tex_image_interp3d(int id, float x, float y, float z, int type, float *result)
{
#if defined(WITH_CLIENT_MPI) || defined(BLENDER_CLIENT)
  client_tex_interp req;
  req.id = id;
  req.x = x;
  req.y = y;
  req.z = z;
  req.type = type;

  // omp_set_lock(&lock_init_data.writelock);
#  pragma omp critical
  {
    // printf("client MPI_Send %d\n", omp_get_thread_num()); fflush(0);
    MPI_Send(&req,
             sizeof(client_tex_interp),
             MPI_BYTE,
             0,
             CLIENT_TAG_CYCLES_tex_interp3d,
             MPI_COMM_WORLD);
    // printf("client MPI_Recv %d\n", omp_get_thread_num()); fflush(0);
    MPI_Recv(result,
             4,
             MPI_FLOAT,
             0,
             CLIENT_TAG_CYCLES_tex_interp3d,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // printf("client finish %d\n", omp_get_thread_num()); fflush(0);
  }
  // omp_unset_lock(&lock_init_data.writelock);
#endif
}

#if 0
void tex_info(char *info, char *mem, size_t size)
{
#  ifdef MPI_REMOTE_TEX
  g_tex_caches.push_back(new TexCache(info, mem, size));
  // g_caches.push_back(new MpiCache(data, (char *)mem, size));
  g_tex_mem_sum += size;
#  else

  // client_kernel_struct *client_data_a = new client_kernel_struct();
  mem_alloc((ccl::device_ptr)mem, size, NULL, NULL /*client_data_a*/);
  //           client_caches.push_back(client_data_a);

  // client_kernel_struct *client_data_c = new client_kernel_struct();
  mem_copy_to((ccl::device_ptr)mem, size, 0, NULL, NULL /*client_data_c*/);
  //          client_caches.push_back(client_data_c);
#  endif
}
#endif

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
  // client_kernel_struct data;
  // get_kernel_data(&data, CLIENT_TAG_CYCLES_tex_free);

  // data.client_mem_data.mem = (host_ptr != NULL) ? host_ptr : (DEVICE_PTR)mem;
  // data.client_mem_data.memSize = memSize;

  // if (_client_data != NULL)
  //  memcpy(_client_data, &data, sizeof(client_kernel_struct));
  // else
  //  send_to_cache(data);

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
/////

void convert_rgb_to_half(unsigned short *destination,
                         unsigned char *source,
                         int tile_h,
                         int tile_w)
{
  tcp::rgb_to_half(destination, source, tile_h, tile_w);
}

#ifdef WITH_CLIENT_OPTIX
void build_optix_bvh(int operation,
                     char *build_input,
                     size_t build_size,
                     int num_motion_steps,
                     char *build_input_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_build_bvh);

  data.client_build_bvh_data.operation = operation;

  data.client_build_bvh_data.build_input = (DEVICE_PTR)build_input;
  data.client_build_bvh_data.build_size = build_size;

  data.client_build_bvh_data.num_motion_steps = num_motion_steps;

  send_to_cache(
      data, (build_input_data == NULL) ? (char *)build_input : build_input_data, build_size);
}
#endif

void close_kernelglobal()
{
}

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_frame_info);

  data.client_frame_info_data.current_frame = current_frame;
  data.client_frame_info_data.current_frame_preview = current_frame_preview;
  data.client_frame_info_data.caching_enabled = caching_enabled;

  send_to_cache(data);

#ifdef WITH_CLIENT_MPI_CACHE
  g_current_frame = data.client_frame_info_data.current_frame;
  g_current_frame_preview = data.client_frame_info_data.current_frame_preview;
  g_caching_enabled = data.client_frame_info_data.caching_enabled;
#endif
}

// CCL_NAMESPACE_END
}  // namespace mpi
}  // namespace kernel
}  // namespace cyclesphi
