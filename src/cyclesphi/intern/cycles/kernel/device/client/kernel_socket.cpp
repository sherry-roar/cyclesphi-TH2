#include "kernel_socket.h"
//#include "kernel_file.h"
#include "kernel_tcp.h"

#include <algorithm>
#include <map>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <vector>

//#include "util_texture.h"
//#include "util_types.h"
#define UCHAR4 uchar4

namespace cyclesphi {
namespace kernel {
namespace socket {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MPI_Request *g_req = NULL;

void bcast(void *data, size_t size)
{
  //#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)
  tcp::send_data_cam((char *)data, size);
  //#endif
}

int count_devices()
{
  //#if defined(WITH_CLIENT_MPI_SOCKET) && !defined(BLENDER_CLIENT)
  return 1;
  //#endif
}

void get_kernel_data(client_kernel_struct *data, int tag)
{
  memset(data, 0, sizeof(client_kernel_struct));
  data->client_tag = tag;
}

////////////////////////////////////////////////////////////////////////////

int g_action = 0;
size_t g_mem_sum = 0;
size_t g_tex_mem_sum = 0;

void send_to_cache(client_kernel_struct &data, void *mem, size_t size)
{
  bcast(&data, sizeof(client_kernel_struct));

  if (mem != NULL)
    bcast(mem, size);
}

void path_to_cache(const char *path)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_path_to_cache);

  strcpy(data.client_const_copy_data.name, path);
  send_to_cache(data);
}

/////////////////////////////////////////////

std::vector<char> receive_path_trace_buffer;
int receive_path_trace_buffer_size = 0;

void recv_decode(char *dmem, int width, int height)
{
  tcp::recv_decode(
      dmem, receive_path_trace_buffer.data(), width, height, receive_path_trace_buffer_size);
}

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

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  tile_w *= 2;  // left+right
  stride *= 2;  // left+right
#endif

  size_t u_char_4 = sizeof(unsigned char) * 4;
  size_t u_half_4 = sizeof(unsigned short) * 4;

  tcp::recv_data_data((char *)&num_samples, sizeof(int), false);

  if (pass_stride_sizeof == u_char_4) {
#ifdef WITH_CLIENT_GPUJPEG
    if (receive_path_trace_buffer.size() != (tile_w * tile_h * sizeof(unsigned char) * 4)) {
      receive_path_trace_buffer.resize(tile_w * tile_h * sizeof(unsigned char) * 4);
    }

    // tcp::recv_gpujpeg((char *)buffer_pixels +
    //                        (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
    //                    (char *)&receive_path_trace_buffer[0],
    //                    tile_w,
    //                    tile_h);

    tcp::recv_data_data((char *)&receive_path_trace_buffer_size, sizeof(int), false);
    tcp::recv_data_data((char *)receive_path_trace_buffer.data(), receive_path_trace_buffer_size);
#else
    tcp::recv_data_data((char *)buffer_pixels +
                            (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                        tile_w * tile_h * pass_stride_sizeof);
#endif
  }
  else if (pass_stride_sizeof == u_half_4) {
#ifdef WITH_CLIENT_GPUJPEG
    if (receive_path_trace_buffer.size() != (tile_w * tile_h * sizeof(unsigned char) * 4)) {
      receive_path_trace_buffer.resize(tile_w * tile_h * sizeof(unsigned char) * 4);
    }

    // tcp::recv_gpujpeg((char *)buffer_pixels +
    //                        (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
    //                    (char *)&receive_path_trace_buffer[0],
    //                    tile_w,
    //                    tile_h);

    tcp::recv_data_data((char *)&receive_path_trace_buffer_size, sizeof(int), false);
    tcp::recv_data_data((char *)receive_path_trace_buffer.data(), receive_path_trace_buffer_size);

#else
    //if (receive_path_trace_buffer.size() != (tile_w * tile_h * sizeof(unsigned short) * 4)) {
    //  receive_path_trace_buffer.resize(tile_w * tile_h * sizeof(unsigned short) * 4);
    //}

    //tcp::recv_data_data((char *)&receive_path_trace_buffer[0], receive_path_trace_buffer.size());

    ////tcp::rgb_to_half((unsigned short *)((char *)buffer_pixels +
    ////                                    (offset + tile_x + tile_y * stride) * pass_stride_sizeof),
    ////                 (unsigned char *)&receive_path_trace_buffer[0],
    ////                 tile_h,
    ////                 tile_w);

    tcp::recv_data_data((char *)buffer_pixels +
                            (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                        tile_w * tile_h * pass_stride_sizeof);
#endif
  }
  else {
    tcp::recv_data_data((char *)buffer_pixels +
                            (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                        tile_w * tile_h * pass_stride_sizeof);
  }

  // recv_data_data((char *)&num_samples, sizeof(int));
  // num_samples = ((int *)buffer_pixels)[0];

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

  while (true) {
    // recv_data((char*) buffer_pixels + (offset + tile_x + tile_y * stride) *
    // pass_stride_sizeof, tile_w * tile_h * pass_stride_sizeof);

    tcp::recv_data_data((char *)buffer_pixels +
                            (offset + tile_x + tile_y * stride) * pass_stride_sizeof,
                        tile_w * tile_h * pass_stride_sizeof);

    int pixel_sample_finished = 0;
    tcp::recv_data_data((char *)&pixel_sample_finished, sizeof(int));

    if (pixel_sample_finished == -1)
      break;

    (*update_progress)(task_bin, tile_bin, num_samples, pixel_sample_finished);
  }
}

//////////////////////////////////////////////////////////////////////

// void path_trace(char *buffer,
//                       char *pixels,
//                       int start_sample,
//                       int num_samples,
//                       int sample_offset,
//                       int tile_x,
//                       int tile_y,
//                       int offset,
//                       int stride,
//                       int tile_h,
//                       int tile_w,
//                       int tile_h2,
//                       int tile_w2,
//                       int pass_stride,
//                       // bool use_load_balancing, int tile_step, int compress,
//                       int has_shadow_catcher,
//                       int num_shaders,
//                       unsigned int kernel_features,
//                       DEVICE_PTR buffer_host_ptr,
//                       DEVICE_PTR pixels_host_ptr)

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

  //    printf("samples: %d/%d\n", start_sample, num_samples);
  //    fflush(0);

  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_path_trace);

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

// void blender_camera(void *mem, size_t mem_size, client_kernel_struct *_client_data)
//{
//  client_kernel_struct data;
//  get_kernel_data(&data, CLIENT_TAG_CYCLES_blender_camera);
//
//  data.client_blender_camera_data.mem = (DEVICE_PTR)mem;
//  data.client_blender_camera_data.size = mem_size;
//  if (_client_data != NULL)
//    memcpy(_client_data, &data, sizeof(client_kernel_struct));
//  else
//    send_to_cache(data, (char *)mem, mem_size);
//}

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

#if 1
  if (_client_data == NULL && tcp::is_error()) {
    tcp::server_close();
    tcp::client_close();

    send_to_cache(data);
  }
#endif
}

void free_kg(client_kernel_struct *_client_data)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_free_kg);
  if (_client_data != NULL)
    memcpy(_client_data, &data, sizeof(client_kernel_struct));
  else
    send_to_cache(data);

  // TCP
#if 1
  if (tcp::is_error()) {
    tcp::server_close();
    tcp::client_close();
  }
#endif
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
void tex_image_interp(int id, float x, float y, float *result)
{
}

void tex_image_interp3d(int id, float x, float y, float z, int type, float *result)
{
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

bool is_error()
{
  return tcp::is_error();
}

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
  client_kernel_struct data;
  get_kernel_data(&data, CLIENT_TAG_CYCLES_frame_info);

  data.client_frame_info_data.current_frame = current_frame;
  data.client_frame_info_data.current_frame_preview = current_frame_preview;
  data.client_frame_info_data.caching_enabled = caching_enabled;

  send_to_cache(data);
}

void set_kernel_globals(char *kg)
{

}

// CCL_NAMESPACE_END
}  // namespace socket
}  // namespace kernel
}  // namespace cyclesphi
