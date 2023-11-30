#ifndef __CYCLESPHI_DATA_H__
#define __CYCLESPHI_DATA_H__

#if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID) || defined(WITH_CLIENT_VRGSTREAM)
namespace cyclesphi {
#endif

typedef struct cyclesphi_cam {
  float transform_inverse_view_matrix[12];

  float lens;
  float clip_start;
  float clip_end;

  float sensor_width;
  float sensor_height;
  int sensor_fit;

  float shift_x;
  float shift_y;

  float interocular_distance;
  float convergence_distance;

  float view_camera_zoom;
  float view_camera_offset[2];
  int use_view_camera;
}cyclesphi_cam;

typedef struct cyclesphi_data {
  int width, height;
  int step_samples;  

  struct cyclesphi_cam cam;

//#if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_CLIENT_ULTRAGRID)
  // camera right
  struct cyclesphi_cam cam_right;
//# endif
  
}cyclesphi_data;

#if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID) || defined(WITH_CLIENT_VRGSTREAM)
}
#endif

#endif
