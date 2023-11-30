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

#include "kernel_camera.h"

#include "BLI_math_rotation.h"
#include "BLI_math_vector.h"

//#include "kernel.h"
#include "kernel/types.h"
#include "util/boundbox.h"
#include "util/array.h"

//#include <vector>

//#include <math.h>

#ifndef M_PI_F
#  define M_PI_F (3.1415926535897932f) /* pi */
#endif
#ifndef M_PI_2_F
#  define M_PI_2_F (1.5707963267948966f) /* pi/2 */
#endif
#ifndef M_2PI_F
#  define M_2PI_F (6.2831853071795864f) /* 2*pi */
#endif

CCL_NAMESPACE_BEGIN

struct Camera {
  /* Specifies an offset for the shutter's time interval. */
  enum MotionPosition {
    /* Shutter opens at the current frame. */
    MOTION_POSITION_START = 0,
    /* Shutter is fully open at the current frame. */
    MOTION_POSITION_CENTER = 1,
    /* Shutter closes at the current frame. */
    MOTION_POSITION_END = 2,

    MOTION_NUM_POSITIONS,
  };

  /* Specifies rolling shutter effect. */
  enum RollingShutterType {
    /* No rolling shutter effect. */
    ROLLING_SHUTTER_NONE = 0,
    /* Sensor is being scanned vertically from top to bottom. */
    ROLLING_SHUTTER_TOP = 1,

    ROLLING_SHUTTER_NUM_TYPES,
  };

  /* Stereo Type */
  enum StereoEye {
    STEREO_NONE,
    STEREO_LEFT,
    STEREO_RIGHT,
  };

  /* motion blur */
  float shuttertime;
  MotionPosition motion_position;
  array<float> shutter_curve;
  size_t shutter_table_offset;

  /* ** Rolling shutter effect. ** */
  /* Defines rolling shutter effect type. */
  RollingShutterType rolling_shutter_type;
  /* Specifies exposure time of scanlines when using
   * rolling shutter effect.
   */
  float rolling_shutter_duration;

  /* depth of field */
  float focaldistance;
  float aperturesize;
  uint blades;
  float bladesrotation;

  /* type */
  CameraType type;
  float fov;

  /* panorama */
  PanoramaType panorama_type;
  float fisheye_fov;
  float fisheye_lens;
  float latitude_min;
  float latitude_max;
  float longitude_min;
  float longitude_max;

  /* panorama stereo */
  StereoEye stereo_eye;
  bool use_spherical_stereo;
  float interocular_distance;
  float convergence_distance;
  bool use_pole_merge;
  float pole_merge_angle_from;
  float pole_merge_angle_to;

  /* anamorphic lens bokeh */
  float aperture_ratio;

  /* sensor */
  float sensorwidth;
  float sensorheight;

  /* clipping */
  float nearclip;
  float farclip;

  /* screen */
  int width, height;
  int resolution;
  BoundBox2D viewplane;
  /* width and height change during preview, so we need these for calculating dice rates. */
  int full_width, full_height;
  /* controls how fast the dicing rate falls off for geometry out side of view */
  float offscreen_dicing_scale;

  /* border */
  BoundBox2D border;
  BoundBox2D viewport_camera_border;

  /* transformation */
  Transform matrix;

  /* motion */
  std::vector<Transform> motion;
  bool use_perspective_motion;
  float fov_pre, fov_post;

  /* computed camera parameters */
  ProjectionTransform screentoworld;
  ProjectionTransform rastertoworld;
  ProjectionTransform ndctoworld;
  Transform cameratoworld;

  ProjectionTransform worldtoraster;
  ProjectionTransform worldtoscreen;
  ProjectionTransform worldtondc;
  Transform worldtocamera;

  ProjectionTransform rastertocamera;
  ProjectionTransform cameratoraster;

  ProjectionTransform full_rastertocamera;

  float3 dx;
  float3 dy;

  float3 full_dx;
  float3 full_dy;

  float3 frustum_right_normal;
  float3 frustum_top_normal;

  /* update */
  bool need_update;
  bool need_device_update;
  bool need_flags_update;
  int previous_need_motion;

  /* Kernel camera data, copied here for dicing. */
  KernelCamera kernel_camera;
  std::vector<DecomposedTransform> kernel_camera_motion;

  Camera()
  {
///////////////////////////////////////////////////
    //NodeType *type = NodeType::add("camera", create);

    shuttertime = 1.0f;

//    static NodeEnum motion_position_enum;
//    motion_position_enum.insert("start", MOTION_POSITION_START);
//    motion_position_enum.insert("center", MOTION_POSITION_CENTER);
//    motion_position_enum.insert("end", MOTION_POSITION_END);
//    SOCKET_ENUM(motion_position, "Motion Position", motion_position_enum, MOTION_POSITION_CENTER);
    motion_position = MOTION_POSITION_CENTER;

//    static NodeEnum rolling_shutter_type_enum;
//    rolling_shutter_type_enum.insert("none", ROLLING_SHUTTER_NONE);
//    rolling_shutter_type_enum.insert("top", ROLLING_SHUTTER_TOP);
//    SOCKET_ENUM(rolling_shutter_type,
//                "Rolling Shutter Type",
//                rolling_shutter_type_enum,
//                ROLLING_SHUTTER_NONE);
    rolling_shutter_type = ROLLING_SHUTTER_NONE;
    //SOCKET_FLOAT(rolling_shutter_duration, "Rolling Shutter Duration", 0.1f);
    rolling_shutter_duration = 0.1f;

    //SOCKET_FLOAT_ARRAY(shutter_curve, "Shutter Curve", array<float>());

    //SOCKET_FLOAT(aperturesize, "Aperture Size", 0.0f);
    aperturesize = 0.0f;
    //SOCKET_FLOAT(focaldistance, "Focal Distance", 10.0f);
    focaldistance = 0.0f;
    //SOCKET_UINT(blades, "Blades", 0);
    blades = 0;
    //SOCKET_FLOAT(bladesrotation, "Blades Rotation", 0.0f);
    bladesrotation = 0.0f;

    //SOCKET_TRANSFORM(matrix, "Matrix", transform_identity());
    //SOCKET_TRANSFORM_ARRAY(motion, "Motion", array<Transform>());

    //SOCKET_FLOAT(aperture_ratio, "Aperture Ratio", 1.0f);
    aperture_ratio = 1.0f;

//    static NodeEnum type_enum;
//    type_enum.insert("perspective", CAMERA_PERSPECTIVE);
//    type_enum.insert("orthograph", CAMERA_ORTHOGRAPHIC);
//    type_enum.insert("panorama", CAMERA_PANORAMA);
//    SOCKET_ENUM(camera_type, "Type", type_enum, CAMERA_PERSPECTIVE);
    //camera_type = CAMERA_PERSPECTIVE;

//    static NodeEnum panorama_type_enum;
//    panorama_type_enum.insert("equirectangular", PANORAMA_EQUIRECTANGULAR);
//    panorama_type_enum.insert("mirrorball", PANORAMA_MIRRORBALL);
//    panorama_type_enum.insert("fisheye_equidistant", PANORAMA_FISHEYE_EQUIDISTANT);
//    panorama_type_enum.insert("fisheye_equisolid", PANORAMA_FISHEYE_EQUISOLID);
//    SOCKET_ENUM(panorama_type, "Panorama Type", panorama_type_enum, PANORAMA_EQUIRECTANGULAR);

    //SOCKET_FLOAT(fisheye_fov, "Fisheye FOV", M_PI_F);
    fisheye_fov = M_PI_F;

    fisheye_lens= 10.5f;
    latitude_min=-M_PI_2_F;
    latitude_max= M_PI_2_F;
    longitude_min= -M_PI_F;
    longitude_max= M_PI_F;
    fov= M_PI_4_F;
    fov_pre=M_PI_4_F;
    fov_post=M_PI_4_F;

//    static NodeEnum stereo_eye_enum;
//    stereo_eye_enum.insert("none", STEREO_NONE);
//    stereo_eye_enum.insert("left", STEREO_LEFT);
//    stereo_eye_enum.insert("right", STEREO_RIGHT);
//    SOCKET_ENUM(stereo_eye, "Stereo Eye", stereo_eye_enum, STEREO_NONE);
    stereo_eye =STEREO_NONE;

    //SOCKET_BOOLEAN(use_spherical_stereo, "Use Spherical Stereo", false);
    use_spherical_stereo = false;

    interocular_distance= 0.065f;
    convergence_distance= 30.0f * 0.065f;

    use_pole_merge=false;
    pole_merge_angle_from= 60.0f * M_PI_F / 180.0f;
    pole_merge_angle_to= 75.0f * M_PI_F / 180.0f;

    sensorwidth=0.036f;
    sensorheight= 0.024f;

    nearclip= 1e-5f;
    farclip= 1e5f;

    viewplane.left= 0;
    viewplane.right= 0;
    viewplane.bottom= 0;
    viewplane.top= 0;

    border.left=0;
    border.right= 0;
    border.bottom= 0;
    border.top= 0;

    viewport_camera_border.left= 0;
    viewport_camera_border.right= 0;
    viewport_camera_border.bottom= 0;
    viewport_camera_border.top= 0;

    offscreen_dicing_scale= 1.0f;

    full_width=1024;
    full_height=512;

    //SOCKET_BOOLEAN(use_perspective_motion, "Use Perspective Motion", false);
    use_perspective_motion = false;
///////////////////////////////////////////////////
    shutter_table_offset = -1;  // TABLE_OFFSET_INVALID;

    width = 1024;
    height = 512;
    resolution = 1;

    use_perspective_motion = false;

    shutter_curve.resize(RAMP_TABLE_SIZE);
    for (int i = 0; i < shutter_curve.size(); ++i) {
      shutter_curve[i] = 1.0f;
    }

    compute_auto_viewplane();

    screentoworld = projection_identity();
    rastertoworld = projection_identity();
    ndctoworld = projection_identity();
    rastertocamera = projection_identity();
    cameratoworld = transform_identity();
    worldtoraster = projection_identity();

    full_rastertocamera = projection_identity();

    dx = make_float3(0.0f, 0.0f, 0.0f);
    dy = make_float3(0.0f, 0.0f, 0.0f);

    need_update = true;
    need_device_update = true;
    need_flags_update = true;
    previous_need_motion = -1;

    memset((void *)&kernel_camera, 0, sizeof(kernel_camera));
  }

  ~Camera()
  {
  }

  void compute_auto_viewplane()
  {
    if (type == CAMERA_PANORAMA) {
      viewplane.left = 0.0f;
      viewplane.right = 1.0f;
      viewplane.bottom = 0.0f;
      viewplane.top = 1.0f;
    }
    else {
      float aspect = (float)width / (float)height;
      if (width >= height) {
        viewplane.left = -aspect;
        viewplane.right = aspect;
        viewplane.bottom = -1.0f;
        viewplane.top = 1.0f;
      }
      else {
        viewplane.left = -1.0f;
        viewplane.right = 1.0f;
        viewplane.bottom = -1.0f / aspect;
        viewplane.top = 1.0f / aspect;
      }
    }
  }

  void update(/*Scene *scene*/)
  {
    // Scene::MotionType need_motion = scene->need_motion();

    // if (previous_need_motion != need_motion) {
    //  /* scene's motion model could have been changed since previous device
    //   * camera update this could happen for example in case when one render
    //   * layer has got motion pass and another not */
    //  need_device_update = true;
    //}

    // if (!need_update)
    //  return;

    /* Full viewport to camera border in the viewport. */
    Transform fulltoborder = transform_from_viewplane(viewport_camera_border);
    Transform bordertofull = transform_inverse(fulltoborder);

    /* ndc to raster */
    Transform ndctoraster = transform_scale(width, height, 1.0f) * bordertofull;
    Transform full_ndctoraster = transform_scale(full_width, full_height, 1.0f) * bordertofull;

    /* raster to screen */
    Transform screentondc = fulltoborder * transform_from_viewplane(viewplane);

    Transform screentoraster = ndctoraster * screentondc;
    Transform rastertoscreen = transform_inverse(screentoraster);
    Transform full_screentoraster = full_ndctoraster * screentondc;
    Transform full_rastertoscreen = transform_inverse(full_screentoraster);

    /* screen to camera */
    ProjectionTransform cameratoscreen;
    if (type == CAMERA_PERSPECTIVE)
      cameratoscreen = projection_perspective(fov, nearclip, farclip);
    else if (type == CAMERA_ORTHOGRAPHIC)
      cameratoscreen = projection_orthographic(nearclip, farclip);
    else
      cameratoscreen = projection_identity();

    ProjectionTransform screentocamera = projection_inverse(cameratoscreen);

    rastertocamera = screentocamera * rastertoscreen;
    full_rastertocamera = screentocamera * full_rastertoscreen;
    cameratoraster = screentoraster * cameratoscreen;

    cameratoworld = matrix;
    screentoworld = cameratoworld * screentocamera;
    rastertoworld = cameratoworld * rastertocamera;
    ndctoworld = rastertoworld * ndctoraster;

    /* note we recompose matrices instead of taking inverses of the above, this
     * is needed to avoid inverting near degenerate matrices that happen due to
     * precision issues with large scenes */
    worldtocamera = transform_inverse(matrix);
    worldtoscreen = cameratoscreen * worldtocamera;
    worldtondc = screentondc * worldtoscreen;
    worldtoraster = ndctoraster * worldtondc;

    /* differentials */
    if (type == CAMERA_ORTHOGRAPHIC) {
      dx = transform_perspective_direction(&rastertocamera, make_float3(1, 0, 0));
      dy = transform_perspective_direction(&rastertocamera, make_float3(0, 1, 0));
      full_dx = transform_perspective_direction(&full_rastertocamera, make_float3(1, 0, 0));
      full_dy = transform_perspective_direction(&full_rastertocamera, make_float3(0, 1, 0));
    }
    else if (type == CAMERA_PERSPECTIVE) {
      dx = transform_perspective(&rastertocamera, make_float3(1, 0, 0)) -
           transform_perspective(&rastertocamera, make_float3(0, 0, 0));
      dy = transform_perspective(&rastertocamera, make_float3(0, 1, 0)) -
           transform_perspective(&rastertocamera, make_float3(0, 0, 0));
      full_dx = transform_perspective(&full_rastertocamera, make_float3(1, 0, 0)) -
                transform_perspective(&full_rastertocamera, make_float3(0, 0, 0));
      full_dy = transform_perspective(&full_rastertocamera, make_float3(0, 1, 0)) -
                transform_perspective(&full_rastertocamera, make_float3(0, 0, 0));
    }
    else {
      dx = make_float3(0.0f, 0.0f, 0.0f);
      dy = make_float3(0.0f, 0.0f, 0.0f);
    }

    dx = transform_direction(&cameratoworld, dx);
    dy = transform_direction(&cameratoworld, dy);
    full_dx = transform_direction(&cameratoworld, full_dx);
    full_dy = transform_direction(&cameratoworld, full_dy);

    if (type == CAMERA_PERSPECTIVE) {
      float3 v = transform_perspective(&full_rastertocamera,
                                       make_float3(full_width, full_height, 1.0f));

      frustum_right_normal = normalize(make_float3(v.z, 0.0f, -v.x));
      frustum_top_normal = normalize(make_float3(0.0f, v.z, -v.y));
    }

    /* Compute kernel camera data. */
    KernelCamera *kcam = &kernel_camera;

    /* store matrices */
    kcam->screentoworld = screentoworld;
    kcam->rastertoworld = rastertoworld;
    kcam->rastertocamera = rastertocamera;
    kcam->cameratoworld = cameratoworld;
    kcam->worldtocamera = worldtocamera;
    kcam->worldtoscreen = worldtoscreen;
    kcam->worldtoraster = worldtoraster;
    kcam->worldtondc = worldtondc;
    kcam->ndctoworld = ndctoworld;

    /* camera motion */
    kcam->num_motion_steps = 0;
    kcam->have_perspective_motion = 0;
    kernel_camera_motion.clear();

    /* Test if any of the transforms are actually different. */
    bool have_motion = false;
    for (size_t i = 0; i < motion.size(); i++) {
      have_motion = have_motion || motion[i] != matrix;
    }

    // if (need_motion == Scene::MOTION_PASS) {
    //  /* TODO(sergey): Support perspective (zoom, fov) motion. */
    //  if (type == CAMERA_PANORAMA) {
    //    if (have_motion) {
    //      kcam->motion_pass_pre = transform_inverse(motion[0]);
    //      kcam->motion_pass_post = transform_inverse(motion[motion.size() - 1]);
    //    }
    //    else {
    //      kcam->motion_pass_pre = kcam->worldtocamera;
    //      kcam->motion_pass_post = kcam->worldtocamera;
    //    }
    //  }
    //  else {
    //    if (have_motion) {
    //      kcam->perspective_pre = cameratoraster * transform_inverse(motion[0]);
    //      kcam->perspective_post = cameratoraster * transform_inverse(motion[motion.size() - 1]);
    //    }
    //    else {
    //      kcam->perspective_pre = worldtoraster;
    //      kcam->perspective_post = worldtoraster;
    //    }
    //  }
    //}
    // else if (need_motion == Scene::MOTION_BLUR) {
    //  if (have_motion) {
    //    kernel_camera_motion.resize(motion.size());
    //    transform_motion_decompose(kernel_camera_motion.data(), motion.data(), motion.size());
    //    kcam->num_motion_steps = motion.size();
    //  }

    //  /* TODO(sergey): Support other types of camera. */
    //  if (use_perspective_motion && type == CAMERA_PERSPECTIVE) {
    //    /* TODO(sergey): Move to an utility function and de-duplicate with
    //     * calculation above.
    //     */
    //    ProjectionTransform screentocamera_pre = projection_inverse(
    //        projection_perspective(fov_pre, nearclip, farclip));
    //    ProjectionTransform screentocamera_post = projection_inverse(
    //        projection_perspective(fov_post, nearclip, farclip));

    //    kcam->perspective_pre = screentocamera_pre * rastertoscreen;
    //    kcam->perspective_post = screentocamera_post * rastertoscreen;
    //    kcam->have_perspective_motion = 1;
    //  }
    //}

    /* depth of field */
    kcam->aperturesize = aperturesize;
    kcam->focaldistance = focaldistance;
    kcam->blades = (blades < 3) ? 0.0f : blades;
    kcam->bladesrotation = bladesrotation;

    /* motion blur */
    kcam->shuttertime = /*(need_motion == Scene::MOTION_BLUR) ? shuttertime :*/ -1.0f;

    /* type */
    kcam->type = type;

    /* anamorphic lens bokeh */
    kcam->inv_aperture_ratio = 1.0f / aperture_ratio;

    /* panorama */
    kcam->panorama_type = panorama_type;
    kcam->fisheye_fov = fisheye_fov;
    kcam->fisheye_lens = fisheye_lens;
    kcam->equirectangular_range = make_float4(longitude_min - longitude_max,
                                              -longitude_min,
                                              latitude_min - latitude_max,
                                              -latitude_min + M_PI_2_F);

    switch (stereo_eye) {
      case STEREO_LEFT:
        kcam->interocular_offset = -interocular_distance * 0.5f;
        break;
      case STEREO_RIGHT:
        kcam->interocular_offset = interocular_distance * 0.5f;
        break;
      case STEREO_NONE:
      default:
        kcam->interocular_offset = 0.0f;
        break;
    }

    kcam->convergence_distance = convergence_distance;
    if (use_pole_merge) {
      kcam->pole_merge_angle_from = pole_merge_angle_from;
      kcam->pole_merge_angle_to = pole_merge_angle_to;
    }
    else {
      kcam->pole_merge_angle_from = -1.0f;
      kcam->pole_merge_angle_to = -1.0f;
    }

    /* sensor size */
    kcam->sensorwidth = sensorwidth;
    kcam->sensorheight = sensorheight;

    /* render size */
    kcam->width = width;
    kcam->height = height;
    //kcam->resolution = resolution;

    /* store differentials */
    kcam->dx = float3_to_float4(dx);
    kcam->dy = float3_to_float4(dy);

    /* clipping */
    kcam->nearclip = nearclip;
    kcam->cliplength = (farclip == FLT_MAX) ? FLT_MAX : farclip - nearclip;

    /* Camera in volume. */
    kcam->is_inside_volume = 0;

    /* Rolling shutter effect */
    kcam->rolling_shutter_type = rolling_shutter_type;
    kcam->rolling_shutter_duration = rolling_shutter_duration;

    /* Set further update flags */
    need_update = false;
    need_device_update = true;
    need_flags_update = true;
    // previous_need_motion = need_motion;
  }
};

struct BlenderCamera {
  float nearclip;
  float farclip;

  CameraType type;
  float ortho_scale;

  float lens;
  float shuttertime;
  Camera::MotionPosition motion_position;
  array<float> shutter_curve;

  Camera::RollingShutterType rolling_shutter_type;
  float rolling_shutter_duration;

  float aperturesize;
  uint apertureblades;
  float aperturerotation;
  float focaldistance;

  float2 shift;
  float2 offset;
  float zoom;

  float2 pixelaspect;

  float aperture_ratio;

  PanoramaType panorama_type;
  float fisheye_fov;
  float fisheye_lens;
  float latitude_min;
  float latitude_max;
  float longitude_min;
  float longitude_max;
  bool use_spherical_stereo;
  float interocular_distance;
  float convergence_distance;
  bool use_pole_merge;
  float pole_merge_angle_from;
  float pole_merge_angle_to;

  enum { AUTO, HORIZONTAL, VERTICAL } sensor_fit;
  float sensor_width;
  float sensor_height;

  int full_width;
  int full_height;

  int render_width;
  int render_height;

  BoundBox2D border;
  BoundBox2D viewport_camera_border;
  BoundBox2D pano_viewplane;
  float pano_aspectratio;

  float passepartout_alpha;

  Transform matrix;

  float offscreen_dicing_scale;

  int motion_steps;
};

struct SpaceView3D {
  float clip_start;
  float clip_end;
  float lens;
};
struct RegionView3D {
  Transform transform_inverse_view_matrix;
  float view_camera_zoom;
  float view_camera_offset[2];
};

static void blender_camera_init(BlenderCamera *bcam,
                                int render_resolution_x,
                                int render_resolution_y)
{
  memset((void *)bcam, 0, sizeof(BlenderCamera));

  bcam->nearclip = 1e-5f;
  bcam->farclip = 1e5f;

  bcam->type = CAMERA_PERSPECTIVE;
  bcam->ortho_scale = 1.0f;

  bcam->lens = 50.0f;
  bcam->shuttertime = 1.0f;

  bcam->rolling_shutter_type = Camera::ROLLING_SHUTTER_NONE;
  bcam->rolling_shutter_duration = 0.1f;

  bcam->aperturesize = 0.0f;
  bcam->apertureblades = 0;
  bcam->aperturerotation = 0.0f;
  bcam->focaldistance = 0.0f;

  bcam->zoom = 1.0f;
  bcam->pixelaspect = make_float2(1.0f, 1.0f);
  bcam->aperture_ratio = 1.0f;

  bcam->sensor_width = 36.0f;
  bcam->sensor_height = 24.0f;
  bcam->sensor_fit = BlenderCamera::AUTO;
  bcam->motion_position = Camera::MOTION_POSITION_CENTER;
  bcam->border.right = 1.0f;
  bcam->border.top = 1.0f;
  bcam->pano_viewplane.right = 1.0f;
  bcam->pano_viewplane.top = 1.0f;
  bcam->viewport_camera_border.right = 1.0f;
  bcam->viewport_camera_border.top = 1.0f;
  bcam->offscreen_dicing_scale = 1.0f;
  bcam->matrix = transform_identity();

  /* render resolution */
  bcam->full_width = render_resolution_x;
  //(b_render);
  bcam->full_height = render_resolution_y;
  //(b_render);
}

struct ObjectCamera {
  float clip_start;
  float clip_end;
  float lens;

  float sensor_width;
  float sensor_height;
  int sensor_fit;

  float shift_x;
  float shift_y;

  //StereoEye stereo_eye;
  float interocular_distance;
  float convergence_distance;

  enum sensor_fit_enum {
    sensor_fit_AUTO = 0,
    sensor_fit_HORIZONTAL = 1,
    sensor_fit_VERTICAL = 2,
  };
};
static void blender_camera_from_object(BlenderCamera *bcam,
                                       /*BL::RenderEngine &b_engine,
                                       BL::Object &b_ob,*/
                                       ObjectCamera &b_camera,
                                       bool skip_panorama = false)
{
  // BL::ID b_ob_data = b_ob.data();

  // if (b_ob_data.is_a(&RNA_Camera)) {
  //  BL::Camera b_camera(b_ob_data);
  //  PointerRNA ccamera = RNA_pointer_get(&b_camera.ptr, "cycles");

  bcam->nearclip = b_camera.clip_start;
  bcam->farclip = b_camera.clip_end;

  // switch (b_camera.type()) {
  //  case BL::Camera::type_ORTHO:
  //    bcam->type = CAMERA_ORTHOGRAPHIC;
  //    break;
  //  case BL::Camera::type_PANO:
  //    if (!skip_panorama)
  //      bcam->type = CAMERA_PANORAMA;
  //    else
  //      bcam->type = CAMERA_PERSPECTIVE;
  //    break;
  //  case BL::Camera::type_PERSP:
  //  default:
  //    bcam->type = CAMERA_PERSPECTIVE;
  //    break;
  //}

  // bcam->panorama_type = (PanoramaType)get_enum(
  //    ccamera, "panorama_type", PANORAMA_NUM_TYPES, PANORAMA_EQUIRECTANGULAR);

  // bcam->fisheye_fov = RNA_float_get(&ccamera, "fisheye_fov");
  // bcam->fisheye_lens = RNA_float_get(&ccamera, "fisheye_lens");
  // bcam->latitude_min = RNA_float_get(&ccamera, "latitude_min");
  // bcam->latitude_max = RNA_float_get(&ccamera, "latitude_max");
  // bcam->longitude_min = RNA_float_get(&ccamera, "longitude_min");
  // bcam->longitude_max = RNA_float_get(&ccamera, "longitude_max");

  // bcam->interocular_distance = b_camera.stereo().interocular_distance();
  //bcam->interocular_distance = b_camera.interocular_distance;
  // if (b_camera.stereo().convergence_mode() == BL::CameraStereoData::convergence_mode_PARALLEL) {
  //  bcam->convergence_distance = FLT_MAX;
  //}
  // else {
  //  bcam->convergence_distance = b_camera.stereo().convergence_distance();
  //bcam->convergence_distance = b_camera.convergence_distance;
  //}
  // bcam->use_spherical_stereo = b_engine.use_spherical_stereo(b_ob);

  // bcam->use_pole_merge = b_camera.stereo().use_pole_merge();
  // bcam->pole_merge_angle_from = b_camera.stereo().pole_merge_angle_from();
  // bcam->pole_merge_angle_to = b_camera.stereo().pole_merge_angle_to();

  // bcam->ortho_scale = b_camera.ortho_scale();

  bcam->lens = b_camera.lens;

  // if (b_camera.dof().use_dof()) {
  //  /* allow f/stop number to change aperture_size but still
  //   * give manual control over aperture radius */
  //  float fstop = b_camera.dof().aperture_fstop();
  //  fstop = max(fstop, 1e-5f);

  //  if (bcam->type == CAMERA_ORTHOGRAPHIC)
  //    bcam->aperturesize = 1.0f / (2.0f * fstop);
  //  else
  //    bcam->aperturesize = (bcam->lens * 1e-3f) / (2.0f * fstop);

  //  bcam->apertureblades = b_camera.dof().aperture_blades();
  //  bcam->aperturerotation = b_camera.dof().aperture_rotation();
  //  bcam->focaldistance = blender_camera_focal_distance(b_engine, b_ob, b_camera, bcam);
  //  bcam->aperture_ratio = b_camera.dof().aperture_ratio();
  //}
  // else {
  /* DOF is turned of for the camera. */
  bcam->aperturesize = 0.0f;
  bcam->apertureblades = 0;
  bcam->aperturerotation = 0.0f;
  bcam->focaldistance = 0.0f;
  bcam->aperture_ratio = 1.0f;
  //}

  bcam->shift.x = b_camera.shift_x;  // b_engine.camera_shift_x(b_ob, bcam->use_spherical_stereo);
  bcam->shift.y = b_camera.shift_y;

  bcam->sensor_width = b_camera.sensor_width;
  bcam->sensor_height = b_camera.sensor_height;

  if (b_camera.sensor_fit == ObjectCamera::sensor_fit_AUTO)
    bcam->sensor_fit = BlenderCamera::AUTO;
  else if (b_camera.sensor_fit == ObjectCamera::sensor_fit_HORIZONTAL)
    bcam->sensor_fit = BlenderCamera::HORIZONTAL;
  else
    bcam->sensor_fit = BlenderCamera::VERTICAL;
  //}
  // else if (b_ob_data.is_a(&RNA_Light)) {
  //  /* Can also look through spot light. */
  //  BL::SpotLight b_light(b_ob_data);
  //  float lens = 16.0f / tanf(b_light.spot_size() * 0.5f);
  //  if (lens > 0.0f) {
  //    bcam->lens = lens;
  //  }
  //}

  // bcam->motion_steps = object_motion_steps(b_ob, b_ob);
}

static void blender_camera_from_view(BlenderCamera *bcam,
                                     // BL::RenderEngine &b_engine,
                                     // BL::Scene &b_scene,
                                     ObjectCamera *b_ob,
                                     SpaceView3D &b_v3d,
                                     RegionView3D &b_rv3d,
                                     int width,
                                     int height,
                                     bool skip_panorama = false)
{
  /* 3d view parameters */
  bcam->nearclip = b_v3d.clip_start;
  bcam->farclip = b_v3d.clip_end;
  bcam->lens = b_v3d.lens;
  // bcam->shuttertime = b_scene.render().motion_blur_shutter();

  // BL::CurveMapping b_shutter_curve(b_scene.render().motion_blur_shutter_curve());
  // curvemapping_to_array(b_shutter_curve, bcam->shutter_curve, RAMP_TABLE_SIZE);

  // if (b_rv3d.view_perspective() == BL::RegionView3D::view_perspective_CAMERA) {
  //  /* camera view */
  //  BL::Object b_ob = (b_v3d.use_local_camera()) ? b_v3d.camera() : b_scene.camera();

  if (b_ob) {
    blender_camera_from_object(bcam, *b_ob, skip_panorama);

    //    if (!skip_panorama && bcam->type == CAMERA_PANORAMA) {
    //      /* in panorama camera view, we map viewplane to camera border */
    //      BoundBox2D view_box, cam_box;

    //      BL::RenderSettings b_render_settings(b_scene.render());
    //      blender_camera_view_subset(b_engine,
    //                                 b_render_settings,
    //                                 b_scene,
    //                                 b_ob,
    //                                 b_v3d,
    //                                 b_rv3d,
    //                                 width,
    //                                 height,
    //                                 &view_box,
    //                                 &cam_box);

    //      bcam->pano_viewplane = view_box.make_relative_to(cam_box);
    //    }
    //    else {
    /* magic zoom formula */
    bcam->zoom = (float)b_rv3d.view_camera_zoom;
    bcam->zoom = (1.41421f + bcam->zoom / 50.0f);
    bcam->zoom *= bcam->zoom;
    bcam->zoom = 2.0f / bcam->zoom;

    /* offset */
    bcam->offset = make_float2(b_rv3d.view_camera_offset[0], b_rv3d.view_camera_offset[1]);
    //    }
  }
  //}
  // else if (b_rv3d.view_perspective() == BL::RegionView3D::view_perspective_ORTHO) {
  //  /* orthographic view */
  //  bcam->farclip *= 0.5f;
  //  bcam->nearclip = -bcam->farclip;

  //  float sensor_size;
  //  if (bcam->sensor_fit == BlenderCamera::VERTICAL)
  //    sensor_size = bcam->sensor_height;
  //  else
  //    sensor_size = bcam->sensor_width;

  //  bcam->type = CAMERA_ORTHOGRAPHIC;
  //  bcam->ortho_scale = b_rv3d.view_distance() * sensor_size / b_v3d.lens();
  //}

  bcam->zoom *= 2.0f;

  /* 3d view transform */
  bcam->matrix =
      b_rv3d
          .transform_inverse_view_matrix;  // transform_inverse(get_transform(b_rv3d.view_matrix()));
}

static void blender_camera_border(BlenderCamera *bcam,
                                  // BL::RenderEngine &b_engine,
                                  // BL::RenderSettings &b_render,
                                  // BL::Scene &b_scene,
                                  SpaceView3D &b_v3d,
                                  RegionView3D &b_rv3d,
                                  int width,
                                  int height)
{
  // bool is_camera_view;

  ///* camera view? */
  // is_camera_view = b_rv3d.view_perspective() == BL::RegionView3D::view_perspective_CAMERA;

  // if (!is_camera_view) {
  //  /* for non-camera view check whether render border is enabled for viewport
  //   * and if so use border from 3d viewport
  //   * assume viewport has got correctly clamped border already
  //   */
  //  if (b_v3d.use_render_border()) {
  //    bcam->border.left = b_v3d.render_border_min_x();
  //    bcam->border.right = b_v3d.render_border_max_x();
  //    bcam->border.bottom = b_v3d.render_border_min_y();
  //    bcam->border.top = b_v3d.render_border_max_y();
  //  }
  //  return;
  //}

  // BL::Object b_ob = (b_v3d.use_local_camera()) ? b_v3d.camera() : b_scene.camera();

  // if (!b_ob)
  //  return;

  ///* Determine camera border inside the viewport. */
  // BoundBox2D full_border;
  // blender_camera_border_subset(b_engine,
  //                             b_render,
  //                             b_scene,
  //                             b_v3d,
  //                             b_rv3d,
  //                             b_ob,
  //                             width,
  //                             height,
  //                             full_border,
  //                             &bcam->viewport_camera_border);

  // if (!b_render.use_border()) {
  //  return;
  //}

  // bcam->border.left = b_render.border_min_x();
  // bcam->border.right = b_render.border_max_x();
  // bcam->border.bottom = b_render.border_min_y();
  // bcam->border.top = b_render.border_max_y();

  ///* Determine viewport subset matching camera border. */
  // blender_camera_border_subset(b_engine,
  //                             b_render,
  //                             b_scene,
  //                             b_v3d,
  //                             b_rv3d,
  //                             b_ob,
  //                             width,
  //                             height,
  //                             bcam->border,
  //                             &bcam->border);
  // bcam->border = bcam->border.clamp();
}

static Transform blender_camera_matrix(const Transform &tfm,
                                       const CameraType type,
                                       const PanoramaType panorama_type)
{
  Transform result;

  if (type == CAMERA_PANORAMA) {
    if (panorama_type == PANORAMA_MIRRORBALL) {
      /* Mirror ball camera is looking into the negative Y direction
       * which matches texture mirror ball mapping.
       */
      result = tfm * make_transform(
                         1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    else {
      /* Make it so environment camera needs to be pointed in the direction
       * of the positive x-axis to match an environment texture, this way
       * it is looking at the center of the texture
       */
      result = tfm * make_transform(
                         0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  else {
    /* note the blender camera points along the negative z-axis */
    result = tfm * transform_scale(1.0f, 1.0f, -1.0f);
  }

  return transform_clear_scale(result);
}

static void blender_camera_viewplane(BlenderCamera *bcam,
                                     int width,
                                     int height,
                                     BoundBox2D *viewplane,
                                     float *aspectratio,
                                     float *sensor_size)
{
  /* dimensions */
  float xratio = (float)width * bcam->pixelaspect.x;
  float yratio = (float)height * bcam->pixelaspect.y;

  /* compute x/y aspect and ratio */
  float xaspect, yaspect;
  bool horizontal_fit;

  /* sensor fitting */
  if (bcam->sensor_fit == BlenderCamera::AUTO) {
    horizontal_fit = (xratio > yratio);
    if (sensor_size != NULL) {
      *sensor_size = bcam->sensor_width;
    }
  }
  else if (bcam->sensor_fit == BlenderCamera::HORIZONTAL) {
    horizontal_fit = true;
    if (sensor_size != NULL) {
      *sensor_size = bcam->sensor_width;
    }
  }
  else {
    horizontal_fit = false;
    if (sensor_size != NULL) {
      *sensor_size = bcam->sensor_height;
    }
  }

  if (horizontal_fit) {
    if (aspectratio != NULL) {
      *aspectratio = xratio / yratio;
    }
    xaspect = *aspectratio;
    yaspect = 1.0f;
  }
  else {
    if (aspectratio != NULL) {
      *aspectratio = yratio / xratio;
    }
    xaspect = 1.0f;
    yaspect = *aspectratio;
  }

  /* modify aspect for orthographic scale */
  if (bcam->type == CAMERA_ORTHOGRAPHIC) {
    xaspect = xaspect * bcam->ortho_scale / (*aspectratio * 2.0f);
    yaspect = yaspect * bcam->ortho_scale / (*aspectratio * 2.0f);
    if (aspectratio != NULL) {
      *aspectratio = bcam->ortho_scale / 2.0f;
    }
  }

  if (bcam->type == CAMERA_PANORAMA) {
    /* set viewplane */
    if (viewplane != NULL) {
      *viewplane = bcam->pano_viewplane;
    }
  }
  else {
    /* set viewplane */
    if (viewplane != NULL) {
      viewplane->left = -xaspect;
      viewplane->right = xaspect;
      viewplane->bottom = -yaspect;
      viewplane->top = yaspect;

      /* zoom for 3d camera view */
      *viewplane = (*viewplane) * bcam->zoom;

      /* modify viewplane with camera shift and 3d camera view offset */
      float dx = 2.0f * (*aspectratio * bcam->shift.x + bcam->offset.x * xaspect * 2.0f);
      float dy = 2.0f * (*aspectratio * bcam->shift.y + bcam->offset.y * yaspect * 2.0f);

      viewplane->left += dx;
      viewplane->right += dx;
      viewplane->bottom += dy;
      viewplane->top += dy;
    }
  }
}

static void blender_camera_sync(
    Camera *cam, BlenderCamera *bcam, int width, int height, const char *viewname
    // PointerRNA *cscene
)
{
  /* copy camera to compare later */
  Camera prevcam = *cam;
  float aspectratio, sensor_size;

  /* viewplane */
  blender_camera_viewplane(bcam, width, height, &cam->viewplane, &aspectratio, &sensor_size);

  cam->width = bcam->full_width;
  cam->height = bcam->full_height;

  cam->full_width = width;
  cam->full_height = height;

  /* panorama sensor */
  if (bcam->type == CAMERA_PANORAMA && bcam->panorama_type == PANORAMA_FISHEYE_EQUISOLID) {
    float fit_xratio = (float)bcam->full_width * bcam->pixelaspect.x;
    float fit_yratio = (float)bcam->full_height * bcam->pixelaspect.y;
    bool horizontal_fit;
    float sensor_size;

    if (bcam->sensor_fit == BlenderCamera::AUTO) {
      horizontal_fit = (fit_xratio > fit_yratio);
      sensor_size = bcam->sensor_width;
    }
    else if (bcam->sensor_fit == BlenderCamera::HORIZONTAL) {
      horizontal_fit = true;
      sensor_size = bcam->sensor_width;
    }
    else { /* vertical */
      horizontal_fit = false;
      sensor_size = bcam->sensor_height;
    }

    if (horizontal_fit) {
      cam->sensorwidth = sensor_size;
      cam->sensorheight = sensor_size * fit_yratio / fit_xratio;
    }
    else {
      cam->sensorwidth = sensor_size * fit_xratio / fit_yratio;
      cam->sensorheight = sensor_size;
    }
  }

  /* clipping distances */
  cam->nearclip = bcam->nearclip;
  cam->farclip = bcam->farclip;

  /* type */
  cam->type = bcam->type;

  /* panorama */
  cam->panorama_type = bcam->panorama_type;
  cam->fisheye_fov = bcam->fisheye_fov;
  cam->fisheye_lens = bcam->fisheye_lens;
  cam->latitude_min = bcam->latitude_min;
  cam->latitude_max = bcam->latitude_max;

  cam->longitude_min = bcam->longitude_min;
  cam->longitude_max = bcam->longitude_max;

  /* panorama stereo */
  cam->interocular_distance = bcam->interocular_distance;
  cam->convergence_distance = bcam->convergence_distance;
  cam->use_spherical_stereo = bcam->use_spherical_stereo;

  if (cam->use_spherical_stereo) {
    if (strcmp(viewname, "left") == 0)
      cam->stereo_eye = Camera::STEREO_LEFT;
    else if (strcmp(viewname, "right") == 0)
      cam->stereo_eye = Camera::STEREO_RIGHT;
    else
      cam->stereo_eye = Camera::STEREO_NONE;
  }

  cam->use_pole_merge = bcam->use_pole_merge;
  cam->pole_merge_angle_from = bcam->pole_merge_angle_from;
  cam->pole_merge_angle_to = bcam->pole_merge_angle_to;

  /* anamorphic lens bokeh */
  cam->aperture_ratio = bcam->aperture_ratio;

  /* perspective */
#ifdef WITH_VRCLIENT_XTAL
  cam->fov = 70.0f * (M_PI_F / 180.0f);
#else
  cam->fov = 2.0f * atanf((0.5f * sensor_size) / bcam->lens / aspectratio);
#endif
  cam->focaldistance = bcam->focaldistance;
  cam->aperturesize = bcam->aperturesize;
  cam->blades = bcam->apertureblades;
  cam->bladesrotation = bcam->aperturerotation;

  /* transform */
  cam->matrix = blender_camera_matrix(bcam->matrix, bcam->type, bcam->panorama_type);
  cam->motion.clear();
  cam->motion.resize(bcam->motion_steps, cam->matrix);
  cam->use_perspective_motion = false;
  cam->shuttertime = bcam->shuttertime;
  cam->fov_pre = cam->fov;
  cam->fov_post = cam->fov;
  cam->motion_position = bcam->motion_position;

  cam->rolling_shutter_type = bcam->rolling_shutter_type;
  cam->rolling_shutter_duration = bcam->rolling_shutter_duration;

  //cam->shutter_curve = bcam->shutter_curve;

  /* border */
  cam->border = bcam->border;
  cam->viewport_camera_border = bcam->viewport_camera_border;

  // bcam->offscreen_dicing_scale = RNA_float_get(cscene, "offscreen_dicing_scale");
  cam->offscreen_dicing_scale = bcam->offscreen_dicing_scale;

  /* set update flag */
  // if (cam->modified(prevcam))
  // cam->tag_update();
  cam->update();
}

void sync_view(Camera *scene_camera,
               ObjectCamera *b_obj,
               SpaceView3D &b_v3d,
               RegionView3D &b_rv3d,
               int width,
               int height)
{
  BlenderCamera bcam;
  // BL::RenderSettings b_render_settings(b_scene.render());
  blender_camera_init(&bcam, width, height);
  blender_camera_from_view(&bcam, /*b_engine, b_scene,*/ b_obj, b_v3d, b_rv3d, width, height);
  blender_camera_border(
      &bcam, /*b_engine, b_render_settings, b_scene,*/ b_v3d, b_rv3d, width, height);
  // PointerRNA cscene = RNA_pointer_get(&b_scene.ptr, "cycles");
  blender_camera_sync(scene_camera, &bcam, width, height, "" /*, &cscene*/);

  ///* dicing camera */
  // BL::Object b_ob = BL::Object(RNA_pointer_get(&cscene, "dicing_camera"));
  // if (b_ob) {
  //  BL::Array<float, 16> b_ob_matrix;
  //  blender_camera_from_object(&bcam, b_engine, b_ob);
  //  b_engine.camera_model_matrix(b_ob, bcam.use_spherical_stereo, b_ob_matrix);
  //  bcam.matrix = get_transform(b_ob_matrix);

  //  blender_camera_sync(scene->dicing_camera, &bcam, width, height, "", &cscene);
  //}
  // else {
  //  *scene->dicing_camera = *scene->camera;
  //}
}

Camera g_scene_camera;

CCL_NAMESPACE_END

namespace cyclesphi {
namespace kernel {

void view_to_kernel_camera(char *kdata, cyclesphi_data *cd)
{
  ccl::SpaceView3D b_v3d;
  b_v3d.clip_start = cd->cam.clip_start;
  b_v3d.clip_end = cd->cam.clip_end;
  b_v3d.lens = cd->cam.lens;

  ccl::ObjectCamera b_obj;
  b_obj.clip_start = cd->cam.clip_start;
  b_obj.clip_end = cd->cam.clip_end;
  b_obj.lens = cd->cam.lens;
  b_obj.sensor_fit = cd->cam.sensor_fit;
  b_obj.sensor_width = cd->cam.sensor_width;
  b_obj.sensor_height = cd->cam.sensor_height;
  b_obj.shift_x = cd->cam.shift_x;
  b_obj.shift_y = cd->cam.shift_y;

//#if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_CLIENT_ULTRAGRID)
//  //b_obj.stereo_eye = STEREO_LEFT;
//  b_obj.interocular_distance = cd->cam.interocular_distance;
//  b_obj.convergence_distance = cd->cam.convergence_distance;
//#endif

  ccl::RegionView3D b_rv3d;
  memcpy(&b_rv3d.transform_inverse_view_matrix,
         cd->cam.transform_inverse_view_matrix,
         sizeof(ccl::Transform));

  //b_rv3d.transform_inverse_view_matrix = ccl::transform_identity();

  b_rv3d.view_camera_zoom = cd->cam.view_camera_zoom;
  b_rv3d.view_camera_offset[0] = cd->cam.view_camera_offset[0];
  b_rv3d.view_camera_offset[1] = cd->cam.view_camera_offset[1];

  //memcpy((char*)&ccl::g_scene_camera.kernel_camera, kcam, sizeof(ccl::KernelCamera));

  ccl::sync_view(&ccl::g_scene_camera,
                 (cd->cam.use_view_camera) ? &b_obj : NULL,
                 b_v3d,
                 b_rv3d,
                 cd->width,
                 cd->height);

  memcpy(&((ccl::KernelData *)kdata)->cam, (char *)&ccl::g_scene_camera.kernel_camera, sizeof(ccl::KernelCamera));
}

void bcam_to_kernel_camera(char *kdata, char *bc, int width, int height)
{
  ccl::BlenderCamera *bcam = (ccl::BlenderCamera *)((ccl::KernelData *)kdata)->cam.blender_camera;
  ccl::blender_camera_sync(&ccl::g_scene_camera, bcam, width, height, "");

  memcpy(&((ccl::KernelData *)kdata)->cam,
         (char *)&ccl::g_scene_camera.kernel_camera,
         sizeof(ccl::KernelCamera));
}

#if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_CLIENT_ULTRAGRID)
void view_to_kernel_camera_right(char *kdata, cyclesphi_data *cd)
{
  ccl::SpaceView3D b_v3d;
  b_v3d.clip_start = cd->cam_right.clip_start;
  b_v3d.clip_end = cd->cam_right.clip_end;
  b_v3d.lens = cd->cam_right.lens;

  ccl::ObjectCamera b_obj;
  b_obj.clip_start = cd->cam_right.clip_start;
  b_obj.clip_end = cd->cam_right.clip_end;
  b_obj.lens = cd->cam_right.lens;
  b_obj.sensor_fit = cd->cam_right.sensor_fit;
  b_obj.sensor_width = cd->cam_right.sensor_width;
  b_obj.sensor_height = cd->cam_right.sensor_height;
  b_obj.shift_x = cd->cam_right.shift_x;
  b_obj.shift_y = cd->cam_right.shift_y;

  //b_obj.stereo_eye = STEREO_RIGHT;
//  b_obj.interocular_distance = cd->cam_right.interocular_distance;
//  b_obj.convergence_distance = cd->cam_right.convergence_distance;

  ccl::RegionView3D b_rv3d;
  memcpy(&b_rv3d.transform_inverse_view_matrix,
         cd->cam_right.transform_inverse_view_matrix,
         sizeof(ccl::Transform));

  b_rv3d.view_camera_zoom = cd->cam_right.view_camera_zoom;
  b_rv3d.view_camera_offset[0] = cd->cam_right.view_camera_offset[0];
  b_rv3d.view_camera_offset[1] = cd->cam_right.view_camera_offset[1];

  ccl::sync_view(&ccl::g_scene_camera,
                 (cd->cam_right.use_view_camera) ? &b_obj : NULL,
                 b_v3d,
                 b_rv3d,
                 cd->width,
                 cd->height);

  memcpy(&((ccl::KernelData *)kdata)->cam, (char *)&ccl::g_scene_camera.kernel_camera, sizeof(ccl::KernelCamera));
}

float right_eye[16] = {
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

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


void bcam_to_kernel_camera_right(char *kdata, char *bc, int width, int height)
{
  ccl::BlenderCamera *bcam = (ccl::BlenderCamera *)((ccl::KernelData *)kdata)->cam.blender_camera;
  
  right_eye[3] = (bcam->interocular_distance != 0.0f) ? bcam->interocular_distance :
                                                        0.065f;  // bcam->interocular_distance;

  mul_m444((float*)&bcam->matrix, right_eye, (float *)&bcam->matrix);

  //bcam->convergence_distance = FLT_MAX;
  //bcam->interocular_distance = 0.065f;
  //bcam->use_spherical_stereo = true;

  ccl::blender_camera_sync(&ccl::g_scene_camera, bcam, width, height, "");

  memcpy(&((ccl::KernelData *)kdata)->cam,
         (char *)&ccl::g_scene_camera.kernel_camera,
         sizeof(ccl::KernelCamera));
}

#endif

}  // namespace kernel
}  // namespace cyclesphi