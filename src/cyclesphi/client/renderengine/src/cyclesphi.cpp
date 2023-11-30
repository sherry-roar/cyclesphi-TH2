#include "client_api.h"
#include "cyclesphi_api.h"
#include "cyclesphi_data.h"

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
#  include "libug.h"

#  ifdef _WIN32
#    define RE_API __declspec(dllexport)
#  else
#    define RE_API
#  endif
#  include "cyclesphi.h"

#  include "vrgstream.h"
#else
#  include "kernel_tcp.h"
#endif

#include <iostream>
#include <omp.h>
#include <string.h>
#include <string>

#include <stdlib.h>

//#pragma comment(lib, "opengl32")
//#pragma comment(lib, "glu32")
//#include <gl/gl.h>
//#include <gl/glu.h>

//#include <GL/glew.h>
//#include <GL/wglew.h>

#if defined(WITH_OPENGL)
//#  pragma comment(lib, "opengl32")
//#  include <GL/glew.h>
//#  include <GL/GL.h>
#  include "glew-mx.h"
//#  include <GL/glut.h>
#endif

//////////////////////////
//#undef WITH_CUDA_GL_INTEROP
#ifdef WITH_CUDA_GL_INTEROP
#  include <cuda_gl_interop.h>
#  include <cuda_runtime.h>
#endif
//#endif
//#include "GL/wglew.h"

//////////////////////////

#if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_VRCLIENT_OPENVR)
#  define DWIDTH ((unsigned long long int)g_width * 2L)
#else
#  define DWIDTH (g_width)
#endif

////////^^^^

unsigned int g_width = 2;
unsigned int g_height = 1;

unsigned char *g_pixels_buf = NULL;  //[3] = {NULL, NULL, NULL};

#ifdef WITH_CUDA_GL_INTEROP
void *g_pixels_buf_d = NULL;     //{NULL, NULL, NULL};
void *g_pixels_buf_gl_d = NULL;  //{NULL, NULL, NULL};
cudaGraphicsResource_t g_cuda_resource = 0;
#endif

#if 0
GLuint g_bufferIds[3];   // IDs of PBO
GLuint g_textureIds[3];  // ID of texture
#endif

typedef int render_callback(int);
render_callback *g_render_callback = NULL;

#if defined(WITH_NVPIPE)
char *g_pixels_compressed = NULL;
#endif

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
RenderPacket g_packet;
ug_receiver *g_ug = NULL;
#else
cyclesphi::cyclesphi_data g_cyclesphi_data;
#endif

double g_previousTime[3] = {0, 0, 0};
int g_frameCount[3] = {0, 0, 0};
char fname[1024];

float g_right_eye = 0.035f;

struct stl_tri;
stl_tri *polys = NULL;
size_t polys_size = 0;

int current_samples = 0;

int active_gpu = 1;

void setupTexture(int eye);
void freeTexture(int eye);

/////////////////////////
#ifdef WITH_CUDA_GL_INTEROP
void check_exit()
{
  //#ifndef _WIN32
  exit(-1);
  //#endif
}

#  define cu_assert(stmt) \
    { \
      CUresult result = stmt; \
      if (result != CUDA_SUCCESS) { \
        char err[1024]; \
        sprintf(err, "CUDA error: %s in %s, line %d", cuewErrorString(result), #stmt, __LINE__); \
        std::string message(err); \
        fprintf(stderr, "%s\n", message.c_str()); \
        check_exit(); \
      } \
    } \
    (void)0

#  define cuda_assert(stmt) \
    { \
      if (stmt != cudaSuccess) { \
        char err[1024]; \
        sprintf(err, \
                "CUDA error: %s: %s in %s, line %d", \
                cudaGetErrorName(stmt), \
                cudaGetErrorString(stmt), \
                #stmt, \
                __LINE__); \
        std::string message(err); \
        fprintf(stderr, "%s\n", message.c_str()); \
        check_exit(); \
      } \
    } \
    (void)0

bool cuda_error_(cudaError_t result, const std::string &stmt)
{
  if (result == cudaSuccess)
    return false;

  char err[1024];
  sprintf(err,
          "CUDA error at %s: %s: %s",
          stmt.c_str(),
          cudaGetErrorName(result),
          cudaGetErrorString(result));
  std::string message(err);
  // string message = string_printf("CUDA error at %s: %s", stmt.c_str(), cuewErrorString(result));
  fprintf(stderr, "%s\n", message.c_str());
  return true;
}

#  define cuda_error(stmt) cuda_error_(stmt, #  stmt)

void cuda_error_message(const std::string &message)
{
  fprintf(stderr, "%s\n", message.c_str());
}

void set_active_gpu()
{
  cuda_assert(cudaSetDevice(0));
}
#endif
/////////////////////////

void displayFPS(int type, int tot_samples = 0)
{
  double currentTime = omp_get_wtime();
  g_frameCount[type]++;

  if (currentTime - g_previousTime[type] >= 3.0) {
    double fps = (double)g_frameCount[type] / (currentTime - g_previousTime[type]);
    if (fps > 0.01) {
      char sTemp[1024];
#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
      unsigned int *samples = &g_packet.frame;
#else
      int *samples = (int *)&g_cyclesphi_data.step_samples;
#endif
      sprintf(sTemp,
              "FPS: %.2f, Total Samples: %d, Samples: : %d, Res: %d x %d",
              fps,
              tot_samples,
              samples[0],
              g_width,
              g_height);
      printf("%s\n", sTemp);
    }
    g_frameCount[type] = 0;
    g_previousTime[type] = omp_get_wtime();
  }
}

//////////////////////////
#if 0
void resize(int width, int height)
{
  if (width == g_width && height == g_height && g_pixels)
    return;

  g_width = width;
  g_height = height;

  if (g_pixels) {
    delete[] g_pixels;
  }

#  if defined(WITH_NVPIPE)
  if (g_pixels_compressed) {
    delete[] g_pixels_compressed;
  }
#  endif

  size_t pix_type_size = SIZE_UCHAR4;

#  ifdef WITH_CLIENT_RENDERENGINE_VR
  pix_type_size *= 2;
#  endif

  g_pixels = new char[width * height * pix_type_size];
  memset(g_pixels, 0, width * height * pix_type_size);

#  if defined(WITH_NVPIPE)
  g_pixels_compressed = new char[width * height * pix_type_size];
  memset(g_pixels_compressed, 0, width * height * pix_type_size);
#  endif

#  ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  g_packet.pix_width_eye = width;
  g_packet.pix_height_eye = height;
#  else
  int *size = (int *)&g_cyclesphi_data.width;
  size[0] = width;
  size[1] = height;
#  endif
}
#endif

void resize(int width, int height)
{
  if (width == g_width && height == g_height && g_pixels_buf)
    return;

  if (g_pixels_buf) {
    // freeTexture(2);

#ifdef WITH_CUDA_GL_INTEROP
    set_active_gpu();
    cuda_assert(cudaFreeHost(g_pixels_buf));
#else
    delete[] g_pixels_buf;
#endif
  }

  g_width = width;
  g_height = height;

#ifdef WITH_CUDA_GL_INTEROP
  set_active_gpu();
  cuda_assert(cudaHostAlloc(&g_pixels_buf, DWIDTH * g_height * SIZE_UCHAR4, cudaHostAllocMapped));
  cuda_assert(cudaHostGetDevicePointer(&g_pixels_buf_d, g_pixels_buf, 0));
#else
  g_pixels_buf = new unsigned char[(size_t)width * height * SIZE_UCHAR4];
  memset(g_pixels_buf, 0, (size_t)width * height * SIZE_UCHAR4);
#endif

  // setupTexture(2);

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  g_packet.pix_width_eye = width;
  g_packet.pix_height_eye = height;
#else
  int *size = (int *)&g_cyclesphi_data.width;
  size[0] = width;
  size[1] = height;
#endif
}

#if 0
int recv_pixels_data()
{
#  ifndef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  size_t pix_type_size = SIZE_UCHAR4;

#    ifdef WITH_CLIENT_RENDERENGINE_VR
  pix_type_size *= 2;
#    endif

#    if defined(WITH_NVPIPE)
  cyclesphi::kernel::tcp::recv_nvpipe(
      (char *)g_pixels_compressed, (char *)g_pixels, DWIDTH, g_height);
#    else
  cyclesphi::kernel::tcp::recv_data_data((char *)g_pixels, DWIDTH * g_height * pix_type_size);

#    endif

  // cyclesphi::kernel::tcp::recv_data_data((char *)&current_samples, sizeof(int));

  // char ack = -1;
  // cyclesphi::kernel::tcp::send_data_cam((char *)&ack, sizeof(char));

  displayFPS(1, current_samples);
#  endif

  return 0;
}

#endif

int recv_pixels_data()
{
#ifndef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB

#  ifdef WITH_CLIENT_XOR_RLE

  size_t recv_size = 0;
  cyclesphi::kernel::tcp::recv_data_data((char *)&recv_size, sizeof(recv_size), false);
  cyclesphi::kernel::tcp::recv_data_data((char *)g_pixels_buf, recv_size - sizeof(recv_size));
  displayFPS(1, current_samples, recv_size);

  char *recv_data = util_xor_rle_to_rgb((char *)g_pixels_buf, g_height, DWIDTH, recv_size);

  memcpy((char *)g_pixels_buf, recv_data, (size_t)g_height * DWIDTH * SIZE_UCHAR4);

#  elif defined(WITH_CLIENT_YUV)
  double t2 = omp_get_wtime();

#    ifndef WITH_CLIENT_RENDERENGINE_SENDER
  cyclesphi::kernel::tcp::recv_data_data((char *)g_pixels_buf,
                                         DWIDTH * g_height + DWIDTH * g_height / 2 /*, false*/);
#    endif

  current_samples = ((int *)g_pixels_buf)[0];
  displayFPS(1, current_samples, DWIDTH * g_height + DWIDTH * g_height / 2);

  double t3 = omp_get_wtime();
#    ifdef WITH_CUDA_GL_INTEROP

  cuda_assert(cudaMemcpy(g_pixels_buf_d,
                         g_pixels_buf,
                         DWIDTH * g_height + DWIDTH * g_height / 2,
                         cudaMemcpyDefault));  // cudaMemcpyDefault cudaMemcpyHostToDevice

#    endif

  double t4 = omp_get_wtime();

  CLIENT_DEBUG_PRINTF3("displayFunc: pix:%f, conv:%f\n", t3 - t2, t4 - t3);
#  else

  double t2 = omp_get_wtime();

#    ifdef WITH_NVPIPE

  cyclesphi::kernel::tcp::recv_nvpipe(
      (char *)g_pixels_bufs[0], (char *)g_pixels_bufs_d[0], DWIDTH, g_height);

  displayFPS(1, current_samples, DWIDTH * g_height * pix_type_size);
  double t3 = omp_get_wtime();

#    else

#      ifndef WITH_CLIENT_RENDERENGINE_SENDER
  // cyclesphi::kernel::tcp::recv_data_data((char *)g_pixels_buf,
  //                                         DWIDTH * g_height * SIZE_UCHAR4 /*, false*/);
  cyclesphi::kernel::tcp::recv_data_data((char *)g_pixels_buf,
                                         g_width * g_height * sizeof(char) * 4 /*, false*/);
#      endif

  // cyclesphi::kernel::tcp::recv_data_data((char*)g_pixels_bufs[1],
  //    g_width * g_height * pix_type_size);

  current_samples = ((int *)g_pixels_buf)[0];
  displayFPS(1, current_samples);

  double t3 = omp_get_wtime();
#      ifdef WITH_CUDA_GL_INTEROP
  if (g_pixels_buf_d != NULL && g_pixels_buf != NULL) {
    set_active_gpu();
    cuda_assert(cudaMemcpy(g_pixels_buf_d,
                           g_pixels_buf,
                           DWIDTH * g_height * SIZE_UCHAR4,
                           cudaMemcpyDefault));  // cudaMemcpyDefault cudaMemcpyHostToDevice

    cuda_assert(cudaMemcpy(g_pixels_buf_gl_d,
                           g_pixels_buf_d,
                           DWIDTH * g_height * SIZE_UCHAR4,
                           cudaMemcpyDeviceToDevice));  // cudaMemcpyDefault cudaMemcpyHostToDevice
  }
#      endif

#    endif
  double t4 = omp_get_wtime();
  // CLIENT_DEBUG_PRINTF3("displayFunc: pix:%f, conv:%f\n", t3 - t2, t4 - t3);
#  endif

#endif

  return 0;
}

int send_cam_data()
{
#ifndef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  cyclesphi::kernel::tcp::send_data_cam((char *)&g_cyclesphi_data,
                                        sizeof(cyclesphi::cyclesphi_data));
  // char ack = 0;
  // cyclesphi::kernel::tcp::recv_data_data((char *)&ack, sizeof(char));
#endif
  return 0;
}

int get_DWIDTH()
{
  return DWIDTH;
}

void client_init(const char *server,
                 int port_cam,
                 int port_data,
                 int w,
                 int h,
                 int step_samples,
                 float right_eye)
{
#ifdef WITH_OPENGL
  // glewInit();
#endif

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  ug_receiver_parameters init_params;
  memset(&init_params, 0, sizeof(ug_receiver_parameters));
  init_params.decompress_to = UG_CUDA_RGBA;
  // UG_CUDA_RGBA;                         // : UG_I420;
  init_params.display = "vrg";  //"gl";// "vrg";
  init_params.sender = server;  // "localhost";

  // init_params.port = port_data;
  init_params.disable_strips = 1;
  init_params.port = port_data;
  init_params.force_gpu_decoding = true;

  if (g_ug == NULL)
    g_ug = ug_receiver_start(&init_params);

#else
#  ifdef WITH_SOCKET_ONLY_DATA
  cyclesphi::kernel::tcp::init_sockets_data(server, port_data);
#  else
  cyclesphi::kernel::tcp::init_sockets_cam(server, port_cam, port_data);
#  endif

  g_cyclesphi_data.step_samples = step_samples;
#endif

  g_right_eye = right_eye;

  resize(w, h);
}

void client_close_connection()
{
#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  if (g_ug != NULL) {
    // ug_receiver_done(g_ug);
    // g_ug = NULL;
  }
#else
  cyclesphi::kernel::tcp::client_close();
  cyclesphi::kernel::tcp::server_close();
#endif
}

void set_camera(void *view_martix,
                float lens,
                float nearclip,
                float farclip,
                float sensor_width,
                float sensor_height,
                int sensor_fit,
                float view_camera_zoom,
                float view_camera_offset0,
                float view_camera_offset1,
                int use_view_camera,
                float shift_x,
                float shift_y)
{
#ifndef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  memcpy(
      (char *)g_cyclesphi_data.cam.transform_inverse_view_matrix, view_martix, sizeof(float) * 12);

  g_cyclesphi_data.cam.lens = lens;
  g_cyclesphi_data.cam.clip_start = nearclip;
  g_cyclesphi_data.cam.clip_end = farclip;

  g_cyclesphi_data.cam.sensor_width = sensor_width;
  g_cyclesphi_data.cam.sensor_height = sensor_height;
  g_cyclesphi_data.cam.sensor_fit = sensor_fit;

  g_cyclesphi_data.cam.view_camera_zoom = view_camera_zoom;
  g_cyclesphi_data.cam.view_camera_offset[0] = view_camera_offset0;
  g_cyclesphi_data.cam.view_camera_offset[1] = view_camera_offset1;
  g_cyclesphi_data.cam.use_view_camera = use_view_camera;
  g_cyclesphi_data.cam.shift_x = shift_x;
  g_cyclesphi_data.cam.shift_y = shift_y;
#endif
}

void set_camera_ug(float left,
                   float right,
                   float top,
                   float bottom,

                   float vx,
                   float vy,
                   float vz,

                   float qx,
                   float qy,
                   float qz,
                   float qw)
{
#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  g_packet.left_projection_fov.left = left;
  g_packet.left_projection_fov.right = right;
  g_packet.left_projection_fov.top = top;
  g_packet.left_projection_fov.bottom = bottom;

  g_packet.left_view_pose.position.x = vx;
  g_packet.left_view_pose.position.y = vy;
  g_packet.left_view_pose.position.z = vz;

  g_packet.left_view_pose.orientation.x = qx;
  g_packet.left_view_pose.orientation.y = qy;
  g_packet.left_view_pose.orientation.z = qz;
  g_packet.left_view_pose.orientation.w = qw;
#endif
}

#ifdef WITH_CLIENT_RENDERENGINE_VR

#  include "BLI_math_rotation.h"
#  include "BLI_math_vector.h"

void mul_m444(float *_res, float *_a, float *_b)
{
  float res[4][4], a[4][4], b[4][4];
  unit_m4(a);
  unit_m4(b);

  for (int i = 0; i < 3; i++) {
    memcpy(a[i], _a + i * 4, sizeof(float) * 4);
    memcpy(b[i], _b + i * 4, sizeof(float) * 4);
  }

  mul_m4_m4m4(res, a, b);

  for (int i = 0; i < 3; i++) {
    memcpy(_res + i * 4, res[i], sizeof(float) * 4);
  }
}

float right_eye44[16] = {
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

#endif

void set_camera_right(void *view_martix,
                      float lens,
                      float nearclip,
                      float farclip,
                      float sensor_width,
                      float sensor_height,
                      int sensor_fit,
                      float view_camera_zoom,
                      float view_camera_offset0,
                      float view_camera_offset1,
                      int use_view_camera,
                      float shift_x,
                      float shift_y)
{
#ifdef WITH_CLIENT_RENDERENGINE_VR

  // memcpy((char *)g_cyclesphi_data.cam_right.transform_inverse_view_matrix,
  //       view_martix,
  //       sizeof(float) * 12);

  right_eye44[3] = g_right_eye;
  mul_m444(
      g_cyclesphi_data.cam_right.transform_inverse_view_matrix, right_eye44, (float *)view_martix);

  g_cyclesphi_data.cam_right.lens = lens;
  g_cyclesphi_data.cam_right.clip_start = nearclip;
  g_cyclesphi_data.cam_right.clip_end = farclip;

  g_cyclesphi_data.cam_right.sensor_width = sensor_width;
  g_cyclesphi_data.cam_right.sensor_height = sensor_height;
  g_cyclesphi_data.cam_right.sensor_fit = sensor_fit;

  g_cyclesphi_data.cam_right.view_camera_zoom = view_camera_zoom;
  g_cyclesphi_data.cam_right.view_camera_offset[0] = view_camera_offset0;
  g_cyclesphi_data.cam_right.view_camera_offset[1] = view_camera_offset1;
  g_cyclesphi_data.cam_right.use_view_camera = use_view_camera;
  g_cyclesphi_data.cam_right.shift_x = shift_x;
  g_cyclesphi_data.cam_right.shift_y = shift_y;

#endif
}

void set_camera_right_ug(float left,
                         float right,
                         float top,
                         float bottom,

                         float vx,
                         float vy,
                         float vz,

                         float qx,
                         float qy,
                         float qz,
                         float qw)
{
}

int get_samples()
{
#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  return g_packet.frame;
#else
  int *samples = (int *)&g_cyclesphi_data.step_samples;
  return samples[0];
#endif
}

int get_current_samples()
{
  return current_samples;
}

int get_renderengine_type()
{
#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
  return 2;
#elif defined(WITH_CLIENT_RENDERENGINE_VR)
  return 1;
#else
  return 0;
#endif
}

void get_pixels(void *pixels)
{
#ifdef WITH_CLIENT_RENDERENGINE_VR
  size_t pix_type_size = SIZE_UCHAR4;
  memcpy(pixels, (char *)g_pixels_buf, g_width * g_height * pix_type_size);
#else
  size_t pix_type_size = sizeof(char) * 4;  // SIZE_UCHAR4;
  memcpy(pixels, (char *)g_pixels_buf, g_width * g_height * pix_type_size);
#endif
}

void get_pixels_right(void *pixels)
{
#ifdef WITH_CLIENT_RENDERENGINE_VR
  size_t pix_type_size = SIZE_UCHAR4;
  memcpy(pixels,
         (char *)g_pixels_buf + (g_width * g_height * pix_type_size),
         g_width * g_height * pix_type_size);
#endif
}

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
void re_init()
{
}

void re_render_frame(char *packet)
{
  memcpy(packet, &g_packet, sizeof(RenderPacket));
  resize(g_packet.pix_width_eye, g_packet.pix_height_eye);
}

void re_submit_frame(char *packet, char *sbs_image_data)
{
  displayFPS(0, current_samples);

  RenderPacket *rp = (RenderPacket *)packet;

  size_t pix_size = DWIDTH * g_height * SIZE_UCHAR4;
  //(DWIDTH * g_height * SIZE_UCHAR4 <
  //                   rp->pix_width_eye * rp->pix_height_eye * 4 * DWIDTH / g_width) ?
  //                      DWIDTH * g_height * SIZE_UCHAR4 :
  //        rp->pix_width_eye * rp->pix_height_eye * 4 * DWIDTH / g_width;
  // resize(2 * rp->pix_width_eye, rp->pix_height_eye);

  // memcpy(g_pixels_buf, sbs_image_data, DWIDTH * g_height * pix_type_size);
#  ifdef WITH_CUDA_GL_INTEROP
  // glfwMakeContextCurrent(g_windows[0]);
  cuda_assert(cudaMemcpy(g_pixels_buf_d, sbs_image_data, pix_size, cudaMemcpyDefault));
#  endif

  if (g_render_callback != NULL) {
    g_render_callback(0);
  }
}

#endif

void cuda_gl_unmap_buffer(unsigned int buffer_id)
{
#ifdef WITH_CUDA_GL_INTEROP
  set_active_gpu();
  cuda_assert(cudaGLUnmapBufferObject(buffer_id));
  cuda_assert(cudaGLUnregisterBufferObject(buffer_id));
#endif
}

// void toOrtho(int eye, int width, int height)
//{
//#if 1
//  // set viewport to be the entire window
//  glViewport(0, 0, (GLsizei)width, (GLsizei)height);
//
//  // set orthographic viewing frustum
//  glMatrixMode(GL_PROJECTION);
//  glLoadIdentity();
//
//  if (eye == 2)
//    glOrtho(0, 1, 0, 1, -1, 1);
//
//  if (eye == 0)
//    glOrtho(0, 0.5, 0, 1, -1, 1);
//
//  if (eye == 1)
//    glOrtho(0.5, 1, 0, 1, -1, 1);
//
//  // switch to modelview matrix in order to set scene
//  glMatrixMode(GL_MODELVIEW);
//  glLoadIdentity();
//  #endif
//}
//
// void freeTexture(int eye)
//{
//#if 1
//  // glfwMakeContextCurrent(g_windows[eye]);
//
//  glDeleteTextures(1, &g_textureIds[eye]);
//  if (eye == 0 || eye == 1)
//    glDeleteBuffers(1, &g_bufferIds[eye]);
//
//  if (eye == 2) {
//    cuda_assert(cudaGLUnmapBufferObject(g_bufferIds[eye]));
//    cuda_assert(cudaGLUnregisterBufferObject(g_bufferIds[eye]));
//
//    glDeleteFramebuffers(1, &g_bufferIds[eye]);
//  }
//#endif
//}

void cuda_gl_map_buffer(unsigned int buffer_id)
{
#ifdef WITH_CUDA_GL_INTEROP
  // cuda_assert(cudaGLRegisterBufferObject(buffer_id));
  // cuda_assert(cudaGLMapBufferObject((void **)&g_pixels_buf_d, buffer_id));

  // g_pixels_buf_d
  set_active_gpu();

  cuda_assert(
      cudaGraphicsGLRegisterBuffer(&g_cuda_resource, buffer_id, cudaGraphicsRegisterFlagsNone));
  size_t bytes;
  cuda_assert(cudaGraphicsMapResources(1, &g_cuda_resource, 0));
  cuda_assert(
      cudaGraphicsResourceGetMappedPointer((void **)&g_pixels_buf_gl_d, &bytes, g_cuda_resource));
#endif
}

//// Setup Texture
// void setupTexture(int eye)
//{
//#if 1
//  // glfwMakeContextCurrent(g_windows[eye]);
//  // int w2 = width * 2;
//
//  GLuint pboIds[1];      // IDs of PBO
//  GLuint textureIds[1];  // ID of texture
//
//  // init 2 texture objects
//  glGenTextures(1, textureIds);
//  g_textureIds[eye] = textureIds[0];
//
//  glBindTexture(GL_TEXTURE_2D, g_textureIds[eye]);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//
//  //#if defined(WITH_CUDA_GL_INTEROP)
//#ifdef WITH_CLIENT_YUV
//  glTexImage2D(
//      GL_TEXTURE_2D, 0, GL_LUMINANCE8, DWIDTH, g_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
//      NULL);
//#else
//  glTexImage2D(GL_TEXTURE_2D,
//               0,
//               GL_RGBA8,
//               (eye == 2) ? DWIDTH : DWIDTH / 2,
//               g_height,
//               0,
//               GL_RGBA,
//               GL_UNSIGNED_BYTE,
//               NULL);
//#endif
//  //#else
//  //	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
//  //(GLvoid*)g_pixels); #endif
//
//  glBindTexture(GL_TEXTURE_2D, 0);
//
//#if 1
//  if (eye == 0 || eye == 1)
//  {
//      // create 2 pixel buffer objects, you need to delete them when program exits.
//      // glBufferData() with NULL pointer reserves only memory space.
//      glGenFramebuffers(1, pboIds);
//      g_bufferIds[eye] = pboIds[0];
//
//      glBindFramebuffer(GL_FRAMEBUFFER, g_bufferIds[eye]);
//
//      // Set "renderedTexture" as our colour attachement #0
//      //glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g_textureIds[eye], 0);
//      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
//      g_textureIds[eye], 0);
//
//      // Set the list of draw buffers.
//      //GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
//      //glDrawBuffers(1, drawBuffers); // "1" is the size of DrawBuffers
//
//      glBindFramebuffer(GL_FRAMEBUFFER, 0);
//  }
//#endif
//  // if (eye == 2)
//  {
//
//    // create 2 pixel buffer objects, you need to delete them when program exits.
//    // glBufferData() with NULL pointer reserves only memory space.
//    glGenBuffers(1, pboIds);
//    g_bufferIds[eye] = pboIds[0];
//
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_bufferIds[eye]);
//
//#if defined(WITH_CUDA_GL_INTEROP)
//    //#ifdef WITH_CLIENT_YUV
//    //      glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)w2 * height, 0, GL_DYNAMIC_COPY);
//    //#else
//    glBufferData(GL_PIXEL_UNPACK_BUFFER,
//                 (size_t)((eye == 2) ? DWIDTH : DWIDTH / 2) * g_height * SIZE_UCHAR4,
//                 0,
//                 GL_DYNAMIC_COPY);
//    //#endif
//    cuda_assert(cudaGLRegisterBufferObject(g_bufferIds[eye]));
//    cuda_assert(cudaGLMapBufferObject((void **)&g_pixels_buf_d, g_bufferIds[eye]));
//#else
//    glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)w2 * height * SIZE_UCHAR4, 0, GL_STREAM_DRAW);
//#endif
//
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//  }
//
//  // Create a texture
//  // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
//  // (GLvoid*)g_pixels);
//
//  //// Set up the texture
//  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//
//  //// Enable textures
//  // glEnable(GL_TEXTURE_2D);
//#endif
//}

#if 0
void gl_render(int eye)
{
#  if 1


  // render to texture
  if (eye == 0 || eye == 1) {
    // glBindFramebuffer(GL_FRAMEBUFFER, g_bufferIds[eye]);
    // toOrtho(g_width/2, g_height);

    // glEnable(GL_MULTISAMPLE);

    // Left Eye
    glBindFramebuffer(GL_FRAMEBUFFER, g_bufferIds[eye]);
    // glViewport(0, 0, g_width / 2, g_height);
    //  vr_RenderScene(vr::Eye_Left);
    toOrtho(eye, DWIDTH / 2, g_height);
  }
  else {

    toOrtho(eye, g_width, g_height);

    glBindTexture(GL_TEXTURE_2D, g_textureIds[2]);

    glBegin(GL_QUADS);

    glTexCoord2d(0.0, 0.0);
    glVertex2d(0.0, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex2d(1, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex2d(1, 1);
    glTexCoord2d(0.0, 1.0);
    glVertex2d(0.0, 1);

    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    return;
  }

  // glfwMakeContextCurrent(g_windows[eye]);

  // bind the texture and PBO
  glBindTexture(GL_TEXTURE_2D, g_textureIds[2]);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_bufferIds[2]);

  // copy pixels from PBO to texture object
  // Use offset instead of ponter.

#    ifdef WITH_CLIENT_YUV
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DWIDTH, g_height, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
#    else
  glTexSubImage2D(GL_TEXTURE_2D,
                  0,
                  0,
                  0,
                  DWIDTH,  //(eye == 2) ? g_width : g_width / 2,
                  g_height,
                  GL_RGBA,
                  GL_UNSIGNED_BYTE,
                  0);
#    endif



  // draw a point with texture
  glBindTexture(GL_TEXTURE_2D, g_textureIds[2]);

  glBegin(GL_QUADS);

    glTexCoord2d(0.0, 0.0);
    glVertex2d(0.0, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex2d(1, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex2d(1, 1);
    glTexCoord2d(0.0, 1.0);
    glVertex2d(0.0, 1);


    glEnd();

  // unbind texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  // render to texture
  if (eye == 0 || eye == 1) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

#  endif
}
#endif

void draw_texture()
{
#if 0
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  //gl_render(0);
  //gl_render(1);
  gl_render(2);
#endif
}

void register_render_callback(void *rc)
{
  g_render_callback = (render_callback *)rc;
}
