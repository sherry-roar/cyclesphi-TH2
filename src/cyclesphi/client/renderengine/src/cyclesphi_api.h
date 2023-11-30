#ifndef __CYCLESPHI_API_H__
#define __CYCLESPHI_API_H__

#if defined(__APPLE__)
#  define CYCLESPHI_EXPORT_DLL
#  define CYCLESPHI_EXPORT_STD
#elif defined(_WIN32)
#  define CYCLESPHI_EXPORT_DLL __declspec(dllexport)
#  define CYCLESPHI_EXPORT_STD __stdcall
#else
#  define CYCLESPHI_EXPORT_DLL
#  define CYCLESPHI_EXPORT_STD
#endif

#ifdef __cplusplus
extern "C" {
#endif

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD resize(int width, int height);
// CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD main_loop();
CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD recv_pixels_data();
CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD send_cam_data();

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD
client_init(const char *server, int port_cam, int port_data, int w, int h, int step_samples, float right_eye);
CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD client_close_connection();

// CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD get_camera(void *view_martix);
CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD set_camera(void *view_martix,
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
                                                          float shift_y);

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD set_camera_ug(float left,
                                                             float right,
                                                             float top,
                                                             float bottom,

                                                             float vx,
                                                             float vy,
                                                             float vz,

                                                             float qx,
                                                             float qy,
                                                             float qz,
                                                             float qw);

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD set_camera_right(void *view_martix,
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
                                                                float shift_y);

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD set_camera_right_ug(float left,
                                                                   float right,
                                                                   float top,
                                                                   float bottom,

                                                                   float vx,
                                                                   float vy,
                                                                   float vz,

                                                                   float qx,
                                                                   float qy,
                                                                   float qz,
                                                                   float qw);

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD get_pixels(void *pixels);
CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD get_pixels_right(void *pixels);

CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD get_samples();
CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD get_current_samples();

CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD get_renderengine_type();
CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD draw_texture();

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD cuda_gl_map_buffer(unsigned int buffer_id);
CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD cuda_gl_unmap_buffer(unsigned int buffer_id);

CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD get_DWIDTH();

CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD register_render_callback(void *rc);

#ifdef __cplusplus
}
#endif
#endif
