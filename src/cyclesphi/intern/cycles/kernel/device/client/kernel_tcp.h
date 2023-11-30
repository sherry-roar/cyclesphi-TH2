#pragma once

#include "client_api.h"
#include <stdlib.h>

#define SOCKET_CONNECTIONS 1

#define TCP_OPTIMIZATION

//#ifndef WITH_CLIENT_VRGSTREAM
namespace cyclesphi {
namespace kernel {
namespace tcp {
//#endif

//#define KERNEL_MPI_SOCKET_BUFSIZE 1024

//#ifdef WITH_CLIENT_MPI_SOCKET

#ifdef __cplusplus
extern "C" {
#endif

void write_data_kernelglobal(void *data, CLIENT_SIZE_T size);
bool read_data_kernelglobal(void *data, CLIENT_SIZE_T size);
void close_kernelglobal();

// bool server_create(int port);
// bool client_create(const char* server_name, int port);

bool is_error();

void init_sockets_cam(const char *server = NULL, int port_cam = 0, int port_data = 0);
void init_sockets_data(const char *server = NULL, int port = 0);

bool client_check();
bool server_check();

void client_close();
void server_close();

#if 0 //def TCP_OPTIMIZATION

void send_data_cam(char *data, CLIENT_SIZE_T size, bool ack = false);
void recv_data_cam(char *data, CLIENT_SIZE_T size, bool ack = false);

void send_data_data(char *data, CLIENT_SIZE_T size, bool ack = false);
void recv_data_data(char *data, CLIENT_SIZE_T size, bool ack = false);

#else

void send_data_cam(char *data, CLIENT_SIZE_T size, bool ack = true);
void recv_data_cam(char *data, CLIENT_SIZE_T size, bool ack = true);

void send_data_data(char *data, CLIENT_SIZE_T size, bool ack = true);
void recv_data_data(char *data, CLIENT_SIZE_T size, bool ack = true);

#endif

#ifdef WITH_NETREFLECTOR
CLIENT_SIZE_T netreflector_send_data(int thread, char *data, CLIENT_SIZE_T size);
CLIENT_SIZE_T netreflector_recv_data(int thread, char *data, CLIENT_SIZE_T size);

CLIENT_SIZE_T netreflector_send_cam(int thread, char *data, CLIENT_SIZE_T size);
CLIENT_SIZE_T netreflector_recv_cam(int thread, char *data, CLIENT_SIZE_T size);
#endif

void send_x264(char *pixels, int width, int height);
void send_nvpipe(DEVICE_PTR dmem, char *pixels, int width, int height);
void recv_nvpipe(char *dmem, char *pixels, int width, int height);

void send_gpujpeg(char *dmem, char *pixels, int width, int height);
void recv_gpujpeg(char *dmem, char *pixels, int width, int height);
void recv_decode(char *dmem, char *pixels, int width, int height, int frame_size);

void rgb_to_yuv_i420(
  unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

void yuv_i420_to_rgb(
  unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

void yuv_i420_to_rgb_half(
  unsigned short* destination, unsigned char* source, int tile_h, int tile_w);

void rgb_to_half(
  unsigned short* destination, unsigned char* source, int tile_h, int tile_w);

#ifdef __cplusplus
}
#endif

//#ifndef WITH_CLIENT_VRGSTREAM
//#endif
}
}  // namespace kernel
}  // namespace cyclesphi
//#endif

