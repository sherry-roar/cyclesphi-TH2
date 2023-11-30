/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "kernel/types.h"

#ifndef BLENDER_CLIENT
#include "graph/node.h"
#endif

#include "util/array.h"
#include "util/boundbox.h"
#include "util/projection.h"
#include "util/transform.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceScene;
class Scene;

/* Camera
 *
 * The camera parameters are quite standard, tested to be both compatible with
 * Renderman, and Blender after remapping.
 */

#ifdef BLENDER_CLIENT
class Camera{
 public:

#else
class Camera : public Node {
 public:
  NODE_DECLARE
#endif

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

#ifdef BLENDER_CLIENT
  ///* motion blur */
  //float shuttertime;
  //MotionPosition motion_position;
  //array<float> shutter_curve;
  //size_t shutter_table_offset;

  ///* ** Rolling shutter effect. ** */
  ///* Defines rolling shutter effect type. */
  //RollingShutterType rolling_shutter_type;
  ///* Specifies exposure time of scanlines when using
  // * rolling shutter effect.
  // */
  //float rolling_shutter_duration;

  ///* depth of field */
  //float focaldistance;
  //float aperturesize;
  //uint blades;
  //float bladesrotation;

  ///* type */
  //CameraType type;
  //float fov;

  ///* panorama */
  //PanoramaType panorama_type;
  //float fisheye_fov;
  //float fisheye_lens;
  //float latitude_min;
  //float latitude_max;
  //float longitude_min;
  //float longitude_max;

  ///* panorama stereo */
  //StereoEye stereo_eye;
  //bool use_spherical_stereo;
  //float interocular_distance;
  //float convergence_distance;
  //bool use_pole_merge;
  //float pole_merge_angle_from;
  //float pole_merge_angle_to;

  ///* anamorphic lens bokeh */
  //float aperture_ratio;

  ///* sensor */
  //float sensorwidth;
  //float sensorheight;

  ///* clipping */
  //float nearclip;
  //float farclip;

  ///* screen */
  //int width, height;
  //int resolution;
  //BoundBox2D viewplane;
  ///* width and height change during preview, so we need these for calculating dice rates. */
  //int full_width, full_height;
  ///* controls how fast the dicing rate falls off for geometry out side of view */
  //float offscreen_dicing_scale;

  ///* border */
  //BoundBox2D border;
  //BoundBox2D viewport_camera_border;

  ///* transformation */
  //Transform matrix;

  ///* motion */
  //std::vector<Transform> motion;
  //bool use_perspective_motion;
  //float fov_pre, fov_post;

  //void set_viewplane_left(int vl)
  //{
  //  viewplane.left = vl;
  //}
  //void set_viewplane_right(int vr)
  //{
  //  viewplane.right = vr;
  //}
  //void set_viewplane_top(int vt)
  //{
  //  viewplane.top = vt;
  //}
  //void set_viewplane_bottom(int vb)
  //{
  //  viewplane.bottom = vb;
  //}

  //void set_full_width(int w)
  //{
  //  width = w;
  //}
  //void set_full_height(int h)
  //{
  //  height = h;
  //}

#  define NODE_SOCKET_API(type_, name) \
    type_ name; \
    void set_##name(type_ value) { \
      name = value; \
    } \
    type_ get_##name() { \
      return name; \
    }

#  define NODE_SOCKET_API_ARRAY(type_, name) \
    type_ name; \
    void set_##name(type_ value) { \
      name = value; \
    } \
    type_ get_##name() { \
      return name; \
    }

#  define NODE_SOCKET_API_STRUCT_MEMBER(type_, name, member) \
    void set_##name##_##member(type_ value) { \
      name.member = value; \
    }

#endif

  /* motion blur */
  NODE_SOCKET_API(float, shuttertime)
  NODE_SOCKET_API(MotionPosition, motion_position)
  NODE_SOCKET_API_ARRAY(array<float>, shutter_curve)
  size_t shutter_table_offset;

  /* ** Rolling shutter effect. ** */
  /* Defines rolling shutter effect type. */
  NODE_SOCKET_API(RollingShutterType, rolling_shutter_type)
  /* Specifies exposure time of scan-lines when using
   * rolling shutter effect.
   */
  NODE_SOCKET_API(float, rolling_shutter_duration)

  /* depth of field */
  NODE_SOCKET_API(float, focaldistance)
  NODE_SOCKET_API(float, aperturesize)
  NODE_SOCKET_API(uint, blades)
  NODE_SOCKET_API(float, bladesrotation)

  /* type */
  NODE_SOCKET_API(CameraType, camera_type)
  NODE_SOCKET_API(float, fov)

  /* panorama */
  NODE_SOCKET_API(PanoramaType, panorama_type)
  NODE_SOCKET_API(float, fisheye_fov)
  NODE_SOCKET_API(float, fisheye_lens)
  NODE_SOCKET_API(float, latitude_min)
  NODE_SOCKET_API(float, latitude_max)
  NODE_SOCKET_API(float, longitude_min)
  NODE_SOCKET_API(float, longitude_max)

  NODE_SOCKET_API(float, fisheye_polynomial_k0)
  NODE_SOCKET_API(float, fisheye_polynomial_k1)
  NODE_SOCKET_API(float, fisheye_polynomial_k2)
  NODE_SOCKET_API(float, fisheye_polynomial_k3)
  NODE_SOCKET_API(float, fisheye_polynomial_k4)

  /* panorama stereo */
  NODE_SOCKET_API(StereoEye, stereo_eye)
  NODE_SOCKET_API(bool, use_spherical_stereo)
  NODE_SOCKET_API(float, interocular_distance)
  NODE_SOCKET_API(float, convergence_distance)
  NODE_SOCKET_API(bool, use_pole_merge)
  NODE_SOCKET_API(float, pole_merge_angle_from)
  NODE_SOCKET_API(float, pole_merge_angle_to)

  /* anamorphic lens bokeh */
  NODE_SOCKET_API(float, aperture_ratio)

  /* sensor */
  NODE_SOCKET_API(float, sensorwidth)
  NODE_SOCKET_API(float, sensorheight)

  /* clipping */
  NODE_SOCKET_API(float, nearclip)
  NODE_SOCKET_API(float, farclip)

  /* screen */
  BoundBox2D viewplane;
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewplane, left)
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewplane, right)
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewplane, bottom)
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewplane, top)

  /* width and height change during preview, so we need these for calculating dice rates. */
  NODE_SOCKET_API(int, full_width)
  NODE_SOCKET_API(int, full_height)
  /* controls how fast the dicing rate falls off for geometry out side of view */
  NODE_SOCKET_API(float, offscreen_dicing_scale)

  /* border */
  BoundBox2D border;
  NODE_SOCKET_API_STRUCT_MEMBER(float, border, left)
  NODE_SOCKET_API_STRUCT_MEMBER(float, border, right)
  NODE_SOCKET_API_STRUCT_MEMBER(float, border, bottom)
  NODE_SOCKET_API_STRUCT_MEMBER(float, border, top)

  BoundBox2D viewport_camera_border;
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewport_camera_border, left)
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewport_camera_border, right)
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewport_camera_border, bottom)
  NODE_SOCKET_API_STRUCT_MEMBER(float, viewport_camera_border, top)

  /* transformation */
  NODE_SOCKET_API(Transform, matrix)

  /* motion */
  NODE_SOCKET_API_ARRAY(array<Transform>, motion)
  NODE_SOCKET_API(bool, use_perspective_motion)
  NODE_SOCKET_API(float, fov_pre)
  NODE_SOCKET_API(float, fov_post)

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
  float3 frustum_left_normal;
  float3 frustum_bottom_normal;

  /* update */
  bool need_device_update;
  bool need_flags_update;
  int previous_need_motion;

  /* Kernel camera data, copied here for dicing. */
  KernelCamera kernel_camera;
  array<DecomposedTransform> kernel_camera_motion;

 //private:
  int width;
  int height;

#ifdef BLENDER_CLIENT
 public:
 // Camera(){};
 // virtual ~Camera(){};
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
    if (camera_type == CAMERA_PERSPECTIVE)
      cameratoscreen = projection_perspective(fov, nearclip, farclip);
    else if (camera_type == CAMERA_ORTHOGRAPHIC)
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
    if (camera_type == CAMERA_ORTHOGRAPHIC) {
      dx = transform_perspective_direction(&rastertocamera, make_float3(1, 0, 0));
      dy = transform_perspective_direction(&rastertocamera, make_float3(0, 1, 0));
      full_dx = transform_perspective_direction(&full_rastertocamera, make_float3(1, 0, 0));
      full_dy = transform_perspective_direction(&full_rastertocamera, make_float3(0, 1, 0));
    }
    else if (camera_type == CAMERA_PERSPECTIVE) {
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

    if (camera_type == CAMERA_PERSPECTIVE) {
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
    kcam->type = camera_type;

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
    // kcam->resolution = resolution;

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
    //need_update = false;
    need_device_update = true;
    need_flags_update = true;
    // previous_need_motion = need_motion;
  };
#else
 public:
  /* functions */
  Camera();
  ~Camera();

  void compute_auto_viewplane();

  void update(Scene *scene);

  void device_update(Device *device, DeviceScene *dscene, Scene *scene);
  void device_update_volume(Device *device, DeviceScene *dscene, Scene *scene);
  void device_free(Device *device, DeviceScene *dscene, Scene *scene);

  /* Public utility functions. */
  BoundBox viewplane_bounds_get();

  /* Calculates the width of a pixel at point in world space. */
  float world_to_raster_size(float3 P);

  /* Motion blur. */
  float motion_time(int step) const;
  int motion_step(float time) const;
  bool use_motion() const;

  void set_screen_size(int width_, int height_);

 private:
  /* Private utility functions. */
  float3 transform_raster_to_world(float raster_x, float raster_y);
#endif
};

CCL_NAMESPACE_END

#endif /* __CAMERA_H__ */
