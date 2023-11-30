#include "kernel_tcp.h"

#if defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_MPI_VRCLIENT) || \
    defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_SOCKET) || \
    defined(WITH_CLIENT_VRGSTREAM)
#  ifdef WITH_ZEROMQ

#    include <cstdlib>
#    include <stdio.h>
#    include <string.h>
#    include <string>
#    include <sys/types.h>
#    include <zmq.h>

#  else

#    include <cstdlib>
#    include <stdio.h>
#    include <string.h>
#    include <sys/types.h>

#    ifdef _WIN32

#      include <iostream>
#      include <winsock2.h>
#      include <ws2tcpip.h>

#      pragma comment(lib, "Ws2_32.lib")
#      pragma comment(lib, "Mswsock.lib")
#      pragma comment(lib, "AdvApi32.lib")

#    else
#      include <arpa/inet.h>
#      include <netdb.h>
#      include <netinet/in.h>
#      include <netinet/tcp.h>
#      include <sys/socket.h>
#      include <unistd.h>
#    endif
#  endif
#endif

#ifdef WITH_NVPIPE
#  include "NvPipe.h"
#  include <cuda_gl_interop.h>
#  include <cuda_runtime.h>
#  include <vector>
#endif

#ifdef WITH_CLIENT_GPUJPEG
#  include <libgpujpeg/gpujpeg_common.h>
#  include <libgpujpeg/gpujpeg_decoder.h>
#  include <libgpujpeg/gpujpeg_encoder.h>
#endif

#include <omp.h>

// RGB
#if 0  // def WITH_NETREFLECTOR
#  define TCP_WIN_SIZE_SEND (2L * 1024L * 1024L)
#  define TCP_WIN_SIZE_RECV (2L * 1024L * 1024L)
#else
#  define TCP_WIN_SIZE_SEND (32L * 1024L * 1024L)
#  define TCP_WIN_SIZE_RECV (32L * 1024L * 1024L)
#endif

#ifdef WITH_SOCKET_UDP
#  define TCP_BLK_SIZE (32000L)
#else

//#ifdef _WIN32
#  define TCP_BLK_SIZE (1L * 1024L * 1024L * 1024L
//#  else
//#    define TCP_BLK_SIZE (16L * 1024L)
//#endif

#  define TCP_MAX_SIZE (128L * 1024L * 1024L)

#endif

// RGB
//#define TCP_WIN_SIZE_SEND (2L * 1024L * 1024L)
//#define TCP_WIN_SIZE_RECV (64L * 1024L)
//
//#define TCP_BLK_SIZE (128L * 1024L)
//#define SOCKET_CONNECTIONS 4

//#define SOCKET_SIMPLE

#ifdef _WIN32
#  define KERNEL_SOCKET_SEND(s, buf, len) send(s, buf, (int)len, 0)
#  define KERNEL_SOCKET_RECV(s, buf, len) recv(s, buf, (int)len, 0)
#else
#  define KERNEL_SOCKET_SEND(s, buf, len) write(s, buf, len)
#  define KERNEL_SOCKET_RECV(s, buf, len) read(s, buf, len)
#endif

//#define PRINT_DEBUG

//#ifndef WITH_CLIENT_VRGSTREAM
namespace cyclesphi {
namespace kernel {
namespace tcp {
//#endif

#if defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_MPI_VRCLIENT) || \
    defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_SOCKET) || \
    defined(WITH_CLIENT_VRGSTREAM)

static int g_server_id_cam[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
static int g_client_id_cam[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

static int g_server_id_data[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
static int g_client_id_data[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

static int g_timeval_sec = 60;
static int g_connection_error = 0;

static sockaddr_in g_client_sockaddr_cam[SOCKET_CONNECTIONS];
static sockaddr_in g_server_sockaddr_cam[SOCKET_CONNECTIONS];

static sockaddr_in g_client_sockaddr_data[SOCKET_CONNECTIONS];
static sockaddr_in g_server_sockaddr_data[SOCKET_CONNECTIONS];

// bool check_socket(int id);
// bool check_socket(int id)
//{
//  int error_code;
//
//#    ifdef WIN32
//  int error_code_size = sizeof(error_code);
//#    else
//  socklen_t error_code_size = sizeof(error_code);
//#    endif
//  getsockopt(id, SOL_SOCKET, SO_ERROR, (char *)&error_code, &error_code_size);
//
//  if (error_code != 0) {
//    printf("getsockopt: %d\n", error_code);
//    fflush(0);
//    return false;
//  }
//
//  return true;
//}

int setsock_tcp_windowsize(int inSock, int inTCPWin, int inSend)
{
#  ifdef SO_SNDBUF
  int rc;
  int newTCPWin;

  // assert( inSock >= 0 );

  if (inTCPWin > 0) {

#    ifdef TCP_WINSHIFT

    /* UNICOS requires setting the winshift explicitly */
    if (inTCPWin > 65535) {
      int winShift = 0;
      int scaledWin = inTCPWin >> 16;
      while (scaledWin > 0) {
        scaledWin >>= 1;
        winShift++;
      }

      /* set TCP window shift */
      rc = setsockopt(inSock, IPPROTO_TCP, TCP_WINSHIFT, (char *)&winShift, sizeof(winShift));
      if (rc < 0) {
        return rc;
      }

      /* Note: you cannot verify TCP window shift, since it returns
       * a structure and not the same integer we use to set it. (ugh) */
    }
#    endif /* TCP_WINSHIFT  */

#    ifdef TCP_RFC1323
    /* On AIX, RFC 1323 extensions can be set system-wide,
     * using the 'no' network options command. But we can also set them
     * per-socket, so let's try just in case. */
    if (inTCPWin > 65535) {
      /* enable RFC 1323 */
      int on = 1;
      rc = setsockopt(inSock, IPPROTO_TCP, TCP_RFC1323, (char *)&on, sizeof(on));
      if (rc < 0) {
        return rc;
      }
    }
#    endif /* TCP_RFC1323 */

    if (!inSend) {
      /* receive buffer -- set
       * note: results are verified after connect() or listen(),
       * since some OS's don't show the corrected value until then. */
      newTCPWin = inTCPWin;
      rc = setsockopt(inSock, SOL_SOCKET, SO_RCVBUF, (char *)&newTCPWin, sizeof(newTCPWin));
    }
    else {
      /* send buffer -- set
       * note: results are verified after connect() or listen(),
       * since some OS's don't show the corrected value until then. */
      newTCPWin = inTCPWin;
      rc = setsockopt(inSock, SOL_SOCKET, SO_SNDBUF, (char *)&newTCPWin, sizeof(newTCPWin));
    }
    if (rc < 0) {
      return rc;
    }
  }
#  endif /* SO_SNDBUF */

  return 0;
} /* end setsock_tcp_windowsize */

bool client_check()
{
  return (g_client_id_cam[0] != -1 && g_client_id_data[0] != -1);
  // check_socket(g_client_id_cam) || check_socket(g_client_id_data);
}

bool server_check()
{
  return (g_server_id_cam[0] != -1 && g_server_id_data[0] != -1);
  // check_socket(g_server_id_cam) || check_socket(g_server_id_data);
}

bool is_error()
{
  return g_connection_error != 0;
}

bool init_wsa();
bool init_wsa()
{
#  ifdef WIN32
  WSADATA wsaData;
  // Request Winsock version 2.2
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    WSACleanup();
    return false;
  }

#  endif
  return true;
}

void close_wsa();
void close_wsa()
{
#  ifdef WIN32
  WSACleanup();
#  endif
}

bool server_create(int port,
                   int &server_id,
                   int &client_id,
                   sockaddr_in &server_sock,
                   sockaddr_in &client_sock,
                   bool only_accept = false);

bool server_create(int port,
                   int &server_id,
                   int &client_id,
                   sockaddr_in &server_sock,
                   sockaddr_in &client_sock,
                   bool only_accept)
{
  if (!only_accept) {
    if (!init_wsa()) {
      return false;
    }

    int type = SOCK_STREAM;
    int protocol = IPPROTO_TCP;

#  ifdef WITH_SOCKET_UDP
    type = SOCK_DGRAM;
    protocol = IPPROTO_UDP;
#  endif

    server_id = socket(AF_INET, type, protocol);

    if (server_id == -1) {
      printf("server_id == -1\n");
      fflush(0);
      return false;
    }

#  if !defined(__MIC__) && !defined(WIN32)
    int enable = 1;
    setsockopt(server_id, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));
#  endif

    // timeval tv;
    // tv.tv_sec = g_timeval_sec;
    // tv.tv_usec = 0;
    // if (setsockopt(server_id, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv, sizeof(tv)) < 0) {
    //  printf("setsockopt == -1\n");
    //  fflush(0);
    //  return false;
    //}

    // sockaddr_in sock_name;
    memset(&server_sock, 0, sizeof(server_sock));
    memset(&client_sock, 0, sizeof(client_sock));
    server_sock.sin_family = AF_INET;
    server_sock.sin_port = htons(port);
    server_sock.sin_addr.s_addr = INADDR_ANY;

    int err_bind = bind(server_id, (sockaddr *)&server_sock, sizeof(server_sock));
    if (err_bind == -1) {
      printf("err_bind == -1\n");
      fflush(0);
      return false;
    }

#  ifdef WITH_SOCKET_UDP
    client_id = server_id;
#  else

    int err_listen = listen(server_id, 1);
    if (err_listen == -1) {
      printf("err_listen == -1\n");
      fflush(0);
      return false;
    }
#    if defined(WITH_SOCKET_ONLY_DATA)
    return true;
#    endif
  }

  sockaddr_in client_info;
  socklen_t addr_len = sizeof(client_info);

  printf("listen on %d\n", port);

  client_id = accept(server_id, (sockaddr *)&client_info, &addr_len);
  if (client_id == -1) {
    printf("client_id == -1\n");
    fflush(0);
    return false;
  }
#  endif

    // printf("accept\n");
    printf("accept on %d <-> %d\n", port, client_info.sin_port);
    // printf("accept on %d <-> %d\n", port, client_id.sin_port);

    fflush(0);

    g_connection_error = 0;

    return true;
  }

  bool client_create(const char *server_name, int port, int &client_id, sockaddr_in &client_sock)
  {
    // printf("connect to %s:%d\n", server_name, port);

    if (!init_wsa()) {
      return false;
    }

    hostent *host = gethostbyname(server_name);
    if (host == NULL) {
      printf("host == NULL\n");
      fflush(0);
      return false;
    }

    int type = SOCK_STREAM;
    int protocol = IPPROTO_TCP;

#  ifdef WITH_SOCKET_UDP
    type = SOCK_DGRAM;
    protocol = IPPROTO_UDP;
#  endif

    client_id = socket(AF_INET, type, protocol);
    if (client_id == -1) {
      printf("client_id == -1\n");
      fflush(0);
      return false;
    }

    // timeval tv;
    // tv.tv_sec = g_timeval_sec;
    // tv.tv_usec = 0;
    // if (setsockopt(6client_id, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv, sizeof(tv)) < 0) {
    //  printf("setsockopt == -1\n");
    //  fflush(0);
    //  return false;
    //}
    // netsh int tcp set global autotuninglevel=normal

#  ifdef TCP_OPTIMIZATION
#    ifdef _WIN32
#      define SIO_TCP_SET_ACK_FREQUENCY _WSAIOW(IOC_VENDOR, 23)
    int freq = 1;
    unsigned long bytes = 0;
    int result = WSAIoctl(
        client_id, SIO_TCP_SET_ACK_FREQUENCY, &freq, sizeof(freq), NULL, 0, &bytes, NULL, NULL);
    int i = 1;
    setsockopt(client_id, IPPROTO_TCP, TCP_NODELAY, (char *)&i, sizeof(i));
#    else
    int i = 1;
    setsockopt(client_id, IPPROTO_TCP, TCP_NODELAY, (char *)&i, sizeof(i));
    i = 1;
    setsockopt(client_id, IPPROTO_TCP, TCP_QUICKACK, (char *)&i, sizeof(i));
#    endif

#    if 1  //! defined(WITH_SOCKET_UDP) && !defined(_WIN32)// && !defined(WITH_NETREFLECTOR)
    setsock_tcp_windowsize(client_id, TCP_WIN_SIZE_SEND, 1);
    setsock_tcp_windowsize(client_id, TCP_WIN_SIZE_RECV, 0);
#    endif
#  endif

    // #ifdef WITH_NETREFLECTOR
    //     int opt = TCP_WIN_SIZE_SEND;
    //     if (setsockopt(client_id, SOL_SOCKET, SO_RCVBUF, (char *)&opt, sizeof(opt)) < 0)
    //     {
    //       printf("setsockopt == -1\n");
    //       fflush(0);
    //       return false;
    //     }

    //     if (setsockopt(client_id, SOL_SOCKET, SO_SNDBUF, (char *)&opt, sizeof(opt)) < 0)
    //     {
    //       printf("setsockopt == -1\n");
    //       fflush(0);
    //       return false;
    //     }

    //     opt = 1;
    //     if (setsockopt(client_id, IPPROTO_TCP, TCP_NODELAY, (char *)&opt, sizeof(opt)) <
    //     0) {
    //       printf("setsockopt == -1\n");
    //       fflush(0);
    //       return false;
    //     }
    // #    endif

    // sockaddr_in client_sock;
    memset(&client_sock, 0, sizeof(client_sock));
    client_sock.sin_family = AF_INET;
    client_sock.sin_port = htons(port);
    memcpy(&(client_sock.sin_addr), host->h_addr, host->h_length);

#  ifndef WITH_SOCKET_UDP

    while (true) {
#    ifdef _WIN32
      Sleep(2);
#    else
      usleep(2000000);
#    endif

      int err_connect = connect(client_id, (sockaddr *)&client_sock, sizeof(client_sock));
      if (client_id == -1) {
        printf("disconnect\n");
        return false;
      }

      if (err_connect == -1) {
        // printf("wait on server %s:%d\n", server_name, port);

        //#      ifdef _WIN32
        //      Sleep(2);
        //#      else
        //      usleep(2000000);
        //#      endif
        continue;
      }
      break;
    }
#  endif

    // printf("connect\n");
    printf("connect to %s:%d\n", server_name, port);
    fflush(0);

    g_connection_error = 0;

    return true;
  }

  void close(int id);
  void close(int id)
  {
#  ifdef WIN32
    closesocket(id);
#  else
  close(id);
#  endif
  }

  void client_close()
  {
#  if 0  // ndef _WIN32
#    pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
#  endif
    for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
      // int tid = omp_get_thread_num();
      close(g_client_id_cam[tid]);
      close(g_client_id_data[tid]);

      g_server_id_cam[tid] = -1;
      g_client_id_cam[tid] = -1;

      g_server_id_data[tid] = -1;
      g_client_id_data[tid] = -1;
    }

    g_connection_error = 0;
  }

  void server_close()
  {
#  if 0  // ndef _WIN32
#    pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
#  endif
    for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
      // int tid = omp_get_thread_num();
      close(g_server_id_cam[tid]);
      close(g_server_id_data[tid]);
    }

    g_connection_error = 0;
  }

  void init_sockets_cam(const char *server, int port_cam, int port_data)
  {
    if (g_client_id_cam[0] == -1) {
      init_wsa();
#  if ((defined(BLENDER_CLIENT) || defined(WITH_CLIENT_MPI_VRCLIENT)) && \
       !defined(WITH_NETREFLECTOR))

      const char *env_p_port_cam = std::getenv("SOCKET_SERVER_PORT_CAM");
      if (port_cam == 0) {
        port_cam = (env_p_port_cam) ? atoi(env_p_port_cam) : 7000;
      }

#    if 0  // ndef _WIN32
#      pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
#    endif
      for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
        // int tid = omp_get_thread_num();
        server_create(port_cam + tid,
                      g_server_id_cam[tid],
                      g_client_id_cam[tid],
                      g_server_sockaddr_cam[tid],
                      g_client_sockaddr_cam[tid]);
      }

#    ifdef WITH_SOCKET_UDP
      char ack = -1;
      recv_data_cam(&ack, sizeof(ack), false);
#    endif

      init_sockets_data(server, port_data);

#  else

    const char *env_p_port_cam = std::getenv("SOCKET_SERVER_PORT_CAM");
    if (port_cam == 0) {
      // port_cam = atoi(env_p_port_cam);
      port_cam = (env_p_port_cam) ? atoi(env_p_port_cam) : 7000;
    }

    const char *env_p_name_cam = std::getenv("SOCKET_SERVER_NAME_CAM");
    char server_temp[1024];
    strcpy(server_temp, "localhost");

    if (env_p_name_cam != NULL) {
      strcpy(server_temp, env_p_name_cam);
    }

    if (server != NULL) {
      strcpy(server_temp, server);
    }

#    if 0  // ndef _WIN32
#      pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
#    endif
    for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
      // int tid = omp_get_thread_num();
      client_create(server_temp, port_cam + tid, g_client_id_cam[tid], g_client_sockaddr_cam[tid]);
    }

#    ifndef WITH_CLIENT_RENDERENGINE_SENDER
    init_sockets_data(server, port_data);
#    endif

#    ifdef WITH_SOCKET_UDP
    char ack = -1;
    send_data_cam(&ack, sizeof(ack), false);
#    endif

#  endif
    }
  }

  void init_sockets_data(const char *server, int port)
  {
    if (g_client_id_data[0] == -1) {
      init_wsa();

#  if (!defined(WITH_SOCKET_ONLY_DATA) && !defined(BLENDER_CLIENT) && \
       !defined(WITH_CLIENT_MPI_VRCLIENT)) || \
      (defined(WITH_SOCKET_ONLY_DATA) && defined(BLENDER_CLIENT))

      const char *env_p_port_data = std::getenv("SOCKET_SERVER_PORT_DATA");
      if (port == 0) {
        // port = atoi(env_p_port_data);
        port = (env_p_port_data) ? atoi(env_p_port_data) : 7001;
      }

      const char *env_p_name_data = std::getenv("SOCKET_SERVER_NAME_DATA");
      char server_temp[1024];
      strcpy(server_temp, "localhost");

      if (env_p_name_data != NULL) {
        strcpy(server_temp, env_p_name_data);
      }

      if (server != NULL) {
        strcpy(server_temp, server);
      }

#    ifdef WITH_SOCKET_ONLY_DATA
      //#ifndef _WIN32
      //#        pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
      //#endif
      for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
        // int tid = i;//omp_get_thread_num();
        //#        pragma omp critical
        client_create(server_temp, port, g_client_id_data[tid], g_client_sockaddr_data[tid]);
      }
#    else
#      if 0  // ndef _WIN32
#        pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
#      endif
      for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
        // int tid = omp_get_thread_num();
        client_create(server_temp, port + tid, g_client_id_data[tid], g_client_sockaddr_data[tid]);
      }
#    endif
      // char ack = -1;
      // send_data_data(&ack, sizeof(ack));

#  else

    const char *env_p_port_data = std::getenv("SOCKET_SERVER_PORT_DATA");
    if (port == 0) {
      // port = atoi(env_p_port_data);
      port = (env_p_port_data) ? atoi(env_p_port_data) : 7001;
    }

#    if defined(WITH_SOCKET_ONLY_DATA)
    server_create(port,
                  g_server_id_data[0],
                  g_client_id_data[0],
                  g_server_sockaddr_data[0],
                  g_client_sockaddr_data[0],
                  false);

    for (int tid = 1; tid < SOCKET_CONNECTIONS; tid++) {
      g_server_id_data[tid] = g_server_id_data[0];
      g_server_sockaddr_data[tid] = g_server_sockaddr_data[0];
    }

    //#        pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
    for (int i = 0; i < SOCKET_CONNECTIONS; i++) {
      int tid = i;  // omp_get_thread_num();
      //#pragma omp critical
      server_create(port,
                    g_server_id_data[tid],
                    g_client_id_data[tid],
                    g_server_sockaddr_data[tid],
                    g_client_sockaddr_data[tid],
                    true);
    }

#    else
#      if 0  // ndef _WIN32
#        pragma omp parallel for num_threads(SOCKET_CONNECTIONS)
#      endif
    for (int tid = 0; tid < SOCKET_CONNECTIONS; tid++) {
      // int tid = omp_get_thread_num();
      server_create(port + tid,
                    g_server_id_data[tid],
                    g_client_id_data[tid],
                    g_server_sockaddr_data[tid],
                    g_client_sockaddr_data[tid]);
    }
    // char ack = -1;
    // recv_data_data(&ack, sizeof(ack));
#    endif
#  endif
    }
  }

#  define DEBUG_PRINT(size)  // printf("%s: %lld\n", __FUNCTION__, size);

  void send_data_cam(char *data, CLIENT_SIZE_T size, bool ack_enabled)
  {
    DEBUG_PRINT(size)

    init_sockets_cam();

    if (is_error())
      return;

    CLIENT_SIZE_T sended_size = 0;

    while (sended_size != size) {
      CLIENT_SIZE_T size_to_send = size - sended_size;
      if (size_to_send > TCP_MAX_SIZE) {
        size_to_send = TCP_MAX_SIZE;
      }

      int temp = KERNEL_SOCKET_SEND(g_client_id_cam[0], (char *)data + sended_size, size_to_send);

      if (temp < 1) {
        g_connection_error = 1;
        break;
      }

      sended_size += temp;
    }

    if (ack_enabled) {
      char ack = 0;
      KERNEL_SOCKET_RECV(g_client_id_cam[0], &ack, 1);
      if (ack != 0) {
        printf("error in send_data_cam\n");
        g_connection_error = 1;
      }
    }
  }

  void send_data_data(char *data, CLIENT_SIZE_T size, bool ack_enabled)
  {
    DEBUG_PRINT(size)

    init_sockets_data();

    if (is_error())
      return;

    CLIENT_SIZE_T sended_size = 0;

    while (sended_size != size) {
      CLIENT_SIZE_T size_to_send = size - sended_size;
      if (size_to_send > TCP_MAX_SIZE) {
        size_to_send = TCP_MAX_SIZE;
      }

      int temp = KERNEL_SOCKET_SEND(g_client_id_data[0], (char *)data + sended_size, size_to_send);

      if (temp < 1) {
        g_connection_error = 1;
        break;
      }

      sended_size += temp;
    }

    if (ack_enabled) {
      char ack = 0;
      KERNEL_SOCKET_RECV(g_client_id_data[0], &ack, 1);
      if (ack != 0) {
        printf("error in g_client_id_data\n");
        g_connection_error = 1;
      }
    }
  }

  void recv_data_cam(char *data, CLIENT_SIZE_T size, bool ack_enabled)
  {
    DEBUG_PRINT(size)

    init_sockets_cam();

    if (is_error())
      return;

    CLIENT_SIZE_T sended_size = 0;

    while (sended_size != size) {
      CLIENT_SIZE_T size_to_send = size - sended_size;
      if (size_to_send > TCP_MAX_SIZE) {
        size_to_send = TCP_MAX_SIZE;
      }

      int temp = KERNEL_SOCKET_RECV(g_client_id_cam[0], (char *)data + sended_size, size_to_send);

      if (temp < 1) {
        g_connection_error = 1;
        break;
      }

      sended_size += temp;
    }

    if (ack_enabled) {
      char ack = 0;
      KERNEL_SOCKET_SEND(g_client_id_cam[0], &ack, 1);
      if (ack != 0) {
        printf("error in g_client_id_cam\n");
        g_connection_error = 1;
      }
    }
  }

  void recv_data_data(char *data, CLIENT_SIZE_T size, bool ack_enabled)
  {
    DEBUG_PRINT(size)

    init_sockets_data();

    if (is_error())
      return;

    CLIENT_SIZE_T sended_size = 0;

    while (sended_size != size) {
      CLIENT_SIZE_T size_to_send = size - sended_size;
      if (size_to_send > TCP_MAX_SIZE) {
        size_to_send = TCP_MAX_SIZE;
      }

      int temp = KERNEL_SOCKET_RECV(g_client_id_data[0], (char *)data + sended_size, size_to_send);

      if (temp < 1) {
        g_connection_error = 1;
        break;
      }

      sended_size += temp;
    }

    if (ack_enabled) {
      char ack = 0;
      KERNEL_SOCKET_SEND(g_client_id_data[0], &ack, 1);
      if (ack != 0) {
        printf("error in g_client_id_data\n");
        g_connection_error = 1;
      }
    }
  }

  // limit UDP 65,507 bytes

  void send_data(char *data, CLIENT_SIZE_T size)
  {
    send_data_data(data, size);
  }

  void recv_data(char *data, CLIENT_SIZE_T size)
  {
    recv_data_data(data, size);
  }

  void close_kernelglobal()
  {
    client_close();
  }

  void write_data_kernelglobal(void *data, CLIENT_SIZE_T size)
  {
    send_data_data((char *)data, size);
  }

  bool read_data_kernelglobal(void *data, CLIENT_SIZE_T size)
  {
    recv_data_cam((char *)data, size);

    return true;
  }

#endif

#ifdef WITH_NETREFLECTOR
  CLIENT_SIZE_T netreflector_send_data(int thread, char *data, CLIENT_SIZE_T size)
  {
    return KERNEL_SOCKET_SEND(g_client_id_data[thread], (char *)data, size);
  }

  CLIENT_SIZE_T netreflector_recv_data(int thread, char *data, CLIENT_SIZE_T size)
  {
    return KERNEL_SOCKET_RECV(g_client_id_data[thread], (char *)data, size);
  }

  CLIENT_SIZE_T netreflector_send_cam(int thread, char *data, CLIENT_SIZE_T size)
  {
    return KERNEL_SOCKET_SEND(g_client_id_cam[thread], (char *)data, size);
  }

  CLIENT_SIZE_T netreflector_recv_cam(int thread, char *data, CLIENT_SIZE_T size)
  {
    return KERNEL_SOCKET_RECV(g_client_id_cam[thread], (char *)data, size);
  }
#endif

#if defined(WITH_X264)
#  include "x264.h"

  x264_t *g_encoder = NULL;
  x264_picture_t g_pic_in;
  x264_picture_t g_pic_out;
  x264_nal_t *g_nals;

  void x264_init(int threads, int width, int height, int fps)
  {
    x264_param_t param;
    x264_param_default_preset(&param, "veryfast", "zerolatency");
    param.i_threads = threads;
    param.i_width = width;
    param.i_height = height;
    param.i_fps_num = fps;
    param.i_fps_den = 1;
    // Intra refres:
    param.i_keyint_max = fps;
    param.b_intra_refresh = 1;
    // Rate control:
    param.rc.i_rc_method = X264_RC_CRF;
    param.rc.f_rf_constant = 25;
    param.rc.f_rf_constant_max = 35;
    // For streaming:
    param.b_repeat_headers = 1;
    param.b_annexb = 1;
    x264_param_apply_profile(&param, "baseline");

    g_encoder = x264_encoder_open(&param);
    x264_picture_alloc(&g_pic_in, X264_CSP_I420, width, height);
    // x264_picture_init(&g_pic_in);

    // g_pic_in.img.i_csp = X264_CSP_I420;

    // g_pic_in.img.i_stride[0] = width * height;
    // g_pic_in.img.i_stride[1] = width * height;
    // g_pic_in.img.i_stride[2] = width * height;
    // g_pic_in.img.i_stride[3] = width * height;
    // g_pic_in.img.i_plane = 4;
  }

  int x264_encode(char *pixels, int width, int height)
  {
    if (g_encoder == NULL)
      x264_init(1, width, height, 60);

    size_t image_size = width * height;
    uint8_t *dst_y = (uint8_t *)pixels;
    uint8_t *dst_u = (uint8_t *)pixels + image_size;
    uint8_t *dst_v = (uint8_t *)pixels + image_size + image_size / 4;

    memcpy(g_pic_in.img.plane[0], dst_y, sizeof(char) * width * height);
    memcpy(g_pic_in.img.plane[1], dst_u, sizeof(char) * width * height / 4);
    memcpy(g_pic_in.img.plane[2], dst_v, sizeof(char) * width * height / 4);
    // memcpy(g_pic_in.img.plane[3], pixels + 3 * width * height, sizeof(char) * width * height);

    // memcpy(g_pic_in.img.plane[0], pixels, sizeof(char) * width * height * 4);
    // g_pic_in.img.plane[0] = (uint8_t *)pixels;
    // g_pic_in.img.plane[1] = (uint8_t *)pixels;
    // g_pic_in.img.plane[2] = (uint8_t *)pixels;
    // g_pic_in.img.plane[3] = (uint8_t *)pixels;
    // g_pic_in.i_pts = 1;
    // memcpy(g_pic_in.img.plane[1], pixels + 1 * width * height, sizeof(char) * width * height);
    // memcpy(g_pic_in.img.plane[2], pixels + 2 * width * height, sizeof(char) * width * height);
    // memcpy(g_pic_in.img.plane[3], pixels + 3 * width * height, sizeof(char) * width * height);

    int i_nals;

    int frame_size = x264_encoder_encode(g_encoder, &g_nals, &i_nals, &g_pic_in, &g_pic_out);
    return frame_size;
  }

#endif

  void send_x264(char *pixels, int width, int height)
  {
#ifdef WITH_X264
    int frame_size = x264_encode(&pixels[0], width, height);

    int *s = (int *)pixels;
    s[0] = frame_size;

    memcpy(pixels + sizeof(int), (char *)g_nals->p_payload, frame_size);

    // send_data((char *)&frame_size, sizeof(int));
    // send_data((char *)g_nals->p_payload, frame_size);
    send_data_data((char *)pixels, sizeof(int) + frame_size);
#endif
  }

#ifdef WITH_NVPIPE
#  include "NvPipe.h"
#  include <cuda_gl_interop.h>
#  include <cuda_runtime.h>
#  include <vector>

  NvPipe *g_decoder = NULL;
  NvPipe *g_encoder = NULL;

  // std::vector<uint8_t> g_compressed;
  // char *g_compressed_d = NULL;

  void nvpipe_decode_init(int width, int height)
  {
    // Create decoder
    g_decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, NVPIPE_H264, width, height);
    // g_compressed.resize(width * height * 4);

    // cudaMalloc(&g_compressed_d, width * height * 4);
  }

  void nvpipe_encode_init(int width, int height)
  {
    // NVPIPE_LOSSY
    g_encoder = NvPipe_CreateEncoder(
        NVPIPE_RGBA32, NVPIPE_H264, NVPIPE_LOSSLESS, 100 * 1000 * 1000, 90, width, height);
  }

  int nvpipe_encode(DEVICE_PTR dmem, char *pixels, int width, int height)
  {
    if (g_encoder == NULL)
      nvpipe_encode_init(width, height);

    uint64_t size = NvPipe_Encode(g_encoder,
                                  (char *)dmem,
                                  width * 4,
                                  (uint8_t *)pixels /* + sizeof(int)*/,
                                  width * height * 4 /*- sizeof(int)*/,
                                  width,
                                  height,
                                  false);

    // int *s = (int *)pixels;
    // s[0] = size;

    if (size == 0)
      printf("%s\n", NvPipe_GetError(g_encoder));

    return size;
  }

  void nvpipe_decode(char *dmem, char *pixels, int frame_size, int width, int height)
  {
    if (g_decoder == NULL)
      nvpipe_decode_init(width, height);

    // cudaMemcpy((char *)dmem, (char *)&pixels[0],
    //           frame_size,
    //           cudaMemcpyKind::cudaMemcpyHostToDevice);
    NvPipe_Decode(g_decoder, (uint8_t *)&dmem[0], frame_size, pixels, width, height);
  }
#endif

  void send_nvpipe(DEVICE_PTR dmem, char *pixels, int width, int height)
  {
#ifdef WITH_NVPIPE
    double t0 = omp_get_wtime();
    int frame_size = nvpipe_encode(dmem, &pixels[0], width, height);
    double t1 = omp_get_wtime();
    send_data_data((char *)&frame_size, sizeof(int), false);
    send_data_data((char *)pixels, frame_size);
    double t2 = omp_get_wtime();
    printf("send_nvpipe: %f, %f\n", t1 - t0, t2 - t1);
    //	char ack = 0;
    //    recv_data_data((char *)&ack, sizeof(char));
#endif
  }

  void recv_nvpipe(char *dmem, char *pixels, int width, int height)
  {
#ifdef WITH_NVPIPE
    int frame_size = 0;
    double t0 = omp_get_wtime();
    recv_data_data((char *)&frame_size, sizeof(int), false);
    recv_data_data((char *)dmem, frame_size);
    double t1 = omp_get_wtime();
    nvpipe_decode(dmem, pixels, frame_size, width, height);
    double t2 = omp_get_wtime();
    printf("recv_nvpipe: %f, %f\n", t1 - t0, t2 - t1);
#endif
  }

  void rgb_to_yuv_i420(unsigned char *destination, unsigned char *source, int tile_h, int tile_w)
  {
    unsigned char *dst_y = destination;
    unsigned char *dst_u = destination + tile_w * tile_h;
    unsigned char *dst_v = destination + tile_w * tile_h + tile_w * tile_h / 4;

#pragma omp parallel for
    for (int y = 0; y < tile_h; y++) {
      for (int x = 0; x < tile_w; x++) {

        int index_src = x + y * tile_w;

        // if (x >= tile_w) {
        //    index_src += tile_h * tile_w;
        //}

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

#if 0  // def WITH_CLIENT_GPUJPEG
  void rgb_to_half(
      unsigned short *destination, unsigned char *source, int tile_h, int tile_w)
  {
#  pragma omp parallel for
    for (int y = 0; y < tile_h; y++) {
      for (int x = 0; x < tile_w; x++) {

        int index_src = x + y * tile_w;
        int index_dst = x + y * tile_w;

        float scale = 1.0f / 255.0f;
        unsigned short *h = &destination[index_dst * 4 + 0];
        unsigned char *f = &source[index_src * 3 + 0];

        for (int i = 0; i < 4; i++) {
          /* optimized float to half for pixels:
           * assumes no negative, no nan, no inf, and sets denormal to 0 */
          union {
            unsigned int i;
            float f;
          } in;
          float fscale = ((i==3) ? 255.0f : f[i]) * scale;
          in.f = (fscale > 0.0f) ? ((fscale < 65504.0f) ? fscale : 65504.0f) : 0.0f;
          int x = in.i;

          int absolute = x & 0x7FFFFFFF;
          int Z = absolute + 0xC8000000;
          int result = (absolute < 0x38800000) ? 0 : Z;
          int rshift = (result >> 13);

          h[i] = (rshift & 0x7FFF);
        }
      }
    }
  }

#else
void rgb_to_half(unsigned short *destination, unsigned char *source, int tile_h, int tile_w)
{
#  pragma omp parallel for
  for (int y = 0; y < tile_h; y++) {
    for (int x = 0; x < tile_w; x++) {

      int index_src = x + y * tile_w;
      int index_dst = x + y * tile_w;

      float scale = 1.0f / 255.0f;
      unsigned short *h = &destination[index_dst * 4 + 0];
      unsigned char *f = &source[index_src * 4 + 0];

      for (int i = 0; i < 4; i++) {
        /* optimized float to half for pixels:
         * assumes no negative, no nan, no inf, and sets denormal to 0 */
        union {
          unsigned int i;
          float f;
        } in;
        float fscale = f[i] * scale;
        in.f = (fscale > 0.0f) ? ((fscale < 65504.0f) ? fscale : 65504.0f) : 0.0f;
        int x = in.i;

        int absolute = x & 0x7FFFFFFF;
        int Z = absolute + 0xC8000000;
        int result = (absolute < 0x38800000) ? 0 : Z;
        int rshift = (result >> 13);

        h[i] = (rshift & 0x7FFF);
      }
    }
  }
}
#endif
  void yuv_i420_to_rgb(unsigned char *destination, unsigned char *source, int tile_h, int tile_w)
  {

    unsigned char *src_y = source;
    unsigned char *src_u = source + tile_w * tile_h;
    unsigned char *src_v = source + tile_w * tile_h + tile_w * tile_h / 4;

#pragma omp parallel for
    for (int y = 0; y < tile_h; y++) {
      for (int x = 0; x < tile_w; x++) {

        int index_dst = x + y * tile_w;

        // if (x >= tile_w) {
        //    index_dst += tile_h * tile_w;
        //}

        unsigned char *r = &destination[index_dst * 4 + 0];
        unsigned char *g = &destination[index_dst * 4 + 1];
        unsigned char *b = &destination[index_dst * 4 + 2];
        unsigned char *a = &destination[index_dst * 4 + 3];

        // Y
        int index_y = x + y * tile_w;
        // dst_y[index_y] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
        unsigned char Y = src_y[index_y];

        // U
        // if (x % 2 == 0 && y % 2 == 0) {
        int index_u = (x / 2) + (y / 2) * tile_w / 2;
        // dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
        //}
        unsigned char U = src_u[index_u];

        // V
        // if (x % 2 == 0 && y % 2 == 0) {
        int index_v = (x / 2) + (y / 2) * tile_w / 2;
        // dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
        //}
        unsigned char V = src_v[index_v];

        unsigned char C = Y - 16;
        unsigned char D = U - 128;
        unsigned char E = V - 128;

        // R = clip((298 * C + 409 * E + 128) >> 8)
        //    G = clip((298 * C - 100 * D - 208 * E + 128) >> 8)
        //    B = clip((298 * C + 516 * D + 128) >> 8)

        *r = (298 * C + 409 * E) >> 8;
        *g = (298 * C - 100 * D - 208 * E) >> 8;
        *b = (298 * C + 516 * D) >> 8;
        *a = 255;
      }
    }
  }

  void yuv_i420_to_rgb_half(
      unsigned short *destination, unsigned char *source, int tile_h, int tile_w)
  {

    unsigned char *src_y = source;
    unsigned char *src_u = source + tile_w * tile_h;
    unsigned char *src_v = source + tile_w * tile_h + tile_w * tile_h / 4;

#pragma omp parallel for
    for (int y = 0; y < tile_h; y++) {
      for (int x = 0; x < tile_w; x++) {

        int index_dst = x + y * tile_w;

        // if (x >= tile_w) {
        //    index_dst += tile_h * tile_w;
        //}

        // unsigned short* r = &destination[index_dst * 4 + 0];
        // unsigned short* g = &destination[index_dst * 4 + 1];
        // unsigned short* b = &destination[index_dst * 4 + 2];
        // unsigned short* a = &destination[index_dst * 4 + 3];

        // Y
        int index_y = x + y * tile_w;
        // dst_y[index_y] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
        unsigned char Y = src_y[index_y];

        // U
        // if (x % 2 == 0 && y % 2 == 0) {
        int index_u = (x / 2) + (y / 2) * (tile_w / 2);
        // dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
        //}
        unsigned char U = src_u[index_u];

        // V
        // if (x % 2 == 0 && y % 2 == 0) {
        int index_v = (x / 2) + (y / 2) * (tile_w / 2);
        // dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
        //}
        unsigned char V = src_v[index_v];

        unsigned char C = Y - 16;
        unsigned char D = U - 128;
        unsigned char E = V - 128;

        // R = clip((298 * C + 409 * E + 128) >> 8)
        //    G = clip((298 * C - 100 * D - 208 * E + 128) >> 8)
        //    B = clip((298 * C + 516 * D + 128) >> 8)

        unsigned char rgba[4];
        rgba[0] = (298 * C + 409 * E) >> 8;
        rgba[1] = (298 * C - 100 * D - 208 * E) >> 8;
        rgba[2] = (298 * C + 516 * D) >> 8;
        rgba[3] = 255;

        //*r = (1.164383 * C + 1.596027 * E) * 65535;
        //*g = (1.164383 * C - (0.391762 * D) - (0.812968 * E)) * 65535;
        //*b = (1.164383 * C + 2.017232 * D) * 65535;
        //*a = 65535;

        float scale = 1.0f / 255.0f;
        unsigned short *h = &destination[index_dst * 4 + 0];
        unsigned char *f = &rgba[0];

        for (int i = 0; i < 4; i++) {
          /* optimized float to half for pixels:
           * assumes no negative, no nan, no inf, and sets denormal to 0 */
          union {
            unsigned int i;
            float f;
          } in;
          float fscale = f[i] * scale;
          in.f = (fscale > 0.0f) ? ((fscale < 65504.0f) ? fscale : 65504.0f) : 0.0f;
          int x = in.i;

          int absolute = x & 0x7FFFFFFF;
          int Z = absolute + 0xC8000000;
          int result = (absolute < 0x38800000) ? 0 : Z;
          int rshift = (result >> 13);

          h[i] = (rshift & 0x7FFF);
        }
      }
    }
  }

#ifdef WITH_CLIENT_GPUJPEG
  gpujpeg_encoder *g_encoder = NULL;
  uint8_t *g_image_compressed;

  int g_compressed_quality = -1; //0-100

  int gpujpeg_encode(int width,
                     int height,
                     uint8_t *input_image,
                     uint8_t *image_compressed,
                     int &image_compressed_size)
  {
    // set default encode parametrs, after calling, parameters can be tuned (eg. quality)
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);

    if (g_compressed_quality == -1) {
      g_compressed_quality = 75;
      const char *compressed_quality_env = getenv("GPUJPEG_QUALITY");
      if(compressed_quality_env != NULL){
        g_compressed_quality = atoi(compressed_quality_env);
      }
      param.quality = g_compressed_quality;
    }

    // here we set image parameters
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = width;
    param_image.height = height;
    param_image.comp_count = 3;
    param_image.color_space = GPUJPEG_YCBCR_BT709;     // GPUJPEG_RGB;
    param_image.pixel_format = GPUJPEG_444_U8_P0P1P2;  // GPUJPEG_420_U8_P0P1P2;

    // create encoder
    if (g_encoder == NULL) {
      if ((g_encoder = gpujpeg_encoder_create(0)) == NULL) {
        return 1;
      }
    }

    struct gpujpeg_encoder_input encoder_input;
    // gpujpeg_encoder_input_set_gpu_image(&encoder_input, input_image);
    gpujpeg_encoder_input_set_image(&encoder_input, input_image);

    // compress the image
    if (gpujpeg_encoder_encode(g_encoder,
                               &param,
                               &param_image,
                               &encoder_input,
                               &g_image_compressed,
                               &image_compressed_size) != 0) {
      return 1;
    }

    return 0;
  }

  gpujpeg_decoder *g_decoder = NULL;
  int gpujpeg_decode(int width,
                     int height,
                     uint8_t *input_image,
                     uint8_t *image_compressed,
                     int &image_compressed_size)
  {
    // create decoder
    if (g_decoder == NULL) {
      if ((g_decoder = gpujpeg_decoder_create(0)) == NULL) {
        return 1;
      }
    }

#  ifdef WITH_VRCLIENT
    gpujpeg_decoder_set_output_format(g_decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012Z);
#  else
  gpujpeg_decoder_set_output_format(
      g_decoder, GPUJPEG_RGB, GPUJPEG_444_U16_P012O /* GPUJPEG_444_U8_P012Z*/);
#  endif
    // set decoder default output destination
    gpujpeg_decoder_output decoder_output;
    // gpujpeg_decoder_output_set_default(&decoder_output);
    // gpujpeg_decoder_output_set_custom(&decoder_output, input_image);
    gpujpeg_decoder_output_set_custom_cuda(&decoder_output, input_image);
    // decoder_output.data = input_image;
    // decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;

    // decompress the image
    uint8_t *image_decompressed = NULL;
    int image_decompressed_size = 0;
    if (gpujpeg_decoder_decode(
            g_decoder, image_compressed, image_compressed_size, &decoder_output) != 0) {
      return 1;
    }

    return 0;
  }
#endif
  void send_gpujpeg(char *dmem, char *pixels, int width, int height)
  {
#ifdef WITH_CLIENT_GPUJPEG
    double t0 = omp_get_wtime();
    int frame_size = 0;
    gpujpeg_encode(width, height, (uint8_t *)dmem, (uint8_t *)pixels, frame_size);
    double t1 = omp_get_wtime();
    send_data_data((char *)&frame_size, sizeof(int), false);
    send_data_data((char *)g_image_compressed, frame_size);
    double t2 = omp_get_wtime();
    // printf("send_gpujpeg: %f, %f\n", t1 - t0, t2 - t1);
#endif
  }

  void recv_gpujpeg(char *dmem, char *pixels, int width, int height)
  {
#ifdef WITH_CLIENT_GPUJPEG
    int frame_size = 0;
    double t0 = omp_get_wtime();
    recv_data_data((char *)&frame_size, sizeof(int), false);
    recv_data_data((char *)pixels, frame_size);
    double t1 = omp_get_wtime();
    gpujpeg_decode(width, height, (uint8_t *)dmem, (uint8_t *)pixels, frame_size);
    double t2 = omp_get_wtime();
    // printf("recv_gpujpeg: %f, %f\n", t1 - t0, t2 - t1);
#endif
  }

  void recv_decode(char *dmem, char *pixels, int width, int height, int frame_size)
  {
#ifdef WITH_CLIENT_GPUJPEG
    gpujpeg_decode(width, height, (uint8_t *)dmem, (uint8_t *)pixels, frame_size);
#endif
  }

  //#ifndef WITH_CLIENT_VRGSTREAM
}  // namespace tcp
}  // namespace kernel
}  // namespace cyclesphi
//#endif
