/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifdef BLENDER_CLIENT
#  include "kernel_camera.h"

#  include "BLI_math_rotation.h"
#  include "BLI_math_vector.h"

//#include "kernel.h"
#  include "kernel/types.h"
#  include "util/array.h"
#  include "util/boundbox.h"

//#include <vector>

//#include <math.h>

#  ifndef M_PI_F
#    define M_PI_F (3.1415926535897932f) /* pi */
#  endif
#  ifndef M_PI_2_F
#    define M_PI_2_F (1.5707963267948966f) /* pi/2 */
#  endif
#  ifndef M_2PI_F
#    define M_2PI_F (6.2831853071795864f) /* 2*pi */
#  endif

#  include "scene/camera.h"

#else

#  include "scene/camera.h"
#  include "scene/scene.h"

#  include "blender/sync.h"
#  include "blender/util.h"

#  include "util/log.h"

#endif

CCL_NAMESPACE_BEGIN

/* Blender Camera Intermediate: we first convert both the offline and 3d view
 * render camera to this, and from there convert to our native camera format. */

struct BlenderCamera {
  float nearclip;
  float farclip;

  CameraType type;
  float ortho_scale;

  float lens;
  float shuttertime;
  MotionPosition motion_position;
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

  float fisheye_polynomial_k0;
  float fisheye_polynomial_k1;
  float fisheye_polynomial_k2;
  float fisheye_polynomial_k3;
  float fisheye_polynomial_k4;

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

#ifdef BLENDER_CLIENT
struct SpaceView3D {
  float _clip_start;
  float _clip_end;
  float _lens;

  float clip_start()
  {
    return _clip_start;
  }
  float clip_end()
  {
    return _clip_end;
  }
  float lens()
  {
    return _lens;
  }
};

struct RegionView3D {
  Transform transform_inverse_view_matrix;
  float view_camera_zoom;
  float view_camera_offset[2];
};

static void blender_camera_init(BlenderCamera *bcam,
                                int render_resolution_x,
                                int render_resolution_y)
#else
static void blender_camera_init(BlenderCamera *bcam, BL::RenderSettings &b_render)
#endif
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
  bcam->focaldistance = 10.0f;

  bcam->zoom = 1.0f;
  bcam->pixelaspect = one_float2();
  bcam->aperture_ratio = 1.0f;

  bcam->sensor_width = 36.0f;
  bcam->sensor_height = 24.0f;
  bcam->sensor_fit = BlenderCamera::AUTO;
  bcam->motion_position = MOTION_POSITION_CENTER;
  bcam->border.right = 1.0f;
  bcam->border.top = 1.0f;
  bcam->viewport_camera_border.right = 1.0f;
  bcam->viewport_camera_border.top = 1.0f;
  bcam->pano_viewplane.right = 1.0f;
  bcam->pano_viewplane.top = 1.0f;
  bcam->pano_aspectratio = 0.0f;
  bcam->passepartout_alpha = 0.5f;
  bcam->offscreen_dicing_scale = 1.0f;
  bcam->matrix = transform_identity();

  /* render resolution */
#ifdef BLENDER_CLIENT
  bcam->render_width = render_resolution_x;
  bcam->render_height = render_resolution_y;
#else
  bcam->render_width = render_resolution_x(b_render);
  bcam->render_height = render_resolution_y(b_render);
#endif
  bcam->full_width = bcam->render_width;
  bcam->full_height = bcam->render_height;
}

#ifndef BLENDER_CLIENT
static float blender_camera_focal_distance(BL::RenderEngine &b_engine,
                                           BL::Object &b_ob,
                                           BL::Camera &b_camera,
                                           BlenderCamera *bcam)
{
  BL::Object b_dof_object = b_camera.dof().focus_object();

  if (!b_dof_object)
    return b_camera.dof().focus_distance();

  Transform dofmat = get_transform(b_dof_object.matrix_world());

  string focus_subtarget = b_camera.dof().focus_subtarget();
  if (b_dof_object.pose() && !focus_subtarget.empty()) {
    BL::PoseBone b_bone = b_dof_object.pose().bones[focus_subtarget];
    if (b_bone) {
      dofmat = dofmat * get_transform(b_bone.matrix());
    }
  }

  /* for dof object, return distance along camera Z direction */
  BL::Array<float, 16> b_ob_matrix;
  b_engine.camera_model_matrix(b_ob, bcam->use_spherical_stereo, b_ob_matrix);
  Transform obmat = transform_clear_scale(get_transform(b_ob_matrix));
  float3 view_dir = normalize(transform_get_column(&obmat, 2));
  float3 dof_dir = transform_get_column(&obmat, 3) - transform_get_column(&dofmat, 3);
  return fabsf(dot(view_dir, dof_dir));
}
#endif

#ifdef BLENDER_CLIENT
struct ObjectCamera {
  float clip_start;
  float clip_end;
  float lens;

  float sensor_width;
  float sensor_height;
  int sensor_fit;

  float shift_x;
  float shift_y;

  // StereoEye stereo_eye;
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
  bcam->nearclip = b_camera.clip_start;
  bcam->farclip = b_camera.clip_end;

  bcam->lens = b_camera.lens;

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
}
#else
static void blender_camera_from_object(BlenderCamera *bcam,
                                       BL::RenderEngine &b_engine,
                                       BL::Object &b_ob,
                                       bool skip_panorama = false)
{
  BL::ID b_ob_data = b_ob.data();

  if (b_ob_data.is_a(&RNA_Camera)) {
    BL::Camera b_camera(b_ob_data);
    PointerRNA ccamera = RNA_pointer_get(&b_camera.ptr, "cycles");

    bcam->nearclip = b_camera.clip_start();
    bcam->farclip = b_camera.clip_end();

    switch (b_camera.type()) {
      case BL::Camera::type_ORTHO:
        bcam->type = CAMERA_ORTHOGRAPHIC;
        break;
      case BL::Camera::type_PANO:
        if (!skip_panorama)
          bcam->type = CAMERA_PANORAMA;
        else
          bcam->type = CAMERA_PERSPECTIVE;
        break;
      case BL::Camera::type_PERSP:
      default:
        bcam->type = CAMERA_PERSPECTIVE;
        break;
    }

    bcam->panorama_type = (PanoramaType)get_enum(
        ccamera, "panorama_type", PANORAMA_NUM_TYPES, PANORAMA_EQUIRECTANGULAR);

    bcam->fisheye_fov = RNA_float_get(&ccamera, "fisheye_fov");
    bcam->fisheye_lens = RNA_float_get(&ccamera, "fisheye_lens");
    bcam->latitude_min = RNA_float_get(&ccamera, "latitude_min");
    bcam->latitude_max = RNA_float_get(&ccamera, "latitude_max");
    bcam->longitude_min = RNA_float_get(&ccamera, "longitude_min");
    bcam->longitude_max = RNA_float_get(&ccamera, "longitude_max");

    bcam->fisheye_polynomial_k0 = RNA_float_get(&ccamera, "fisheye_polynomial_k0");
    bcam->fisheye_polynomial_k1 = RNA_float_get(&ccamera, "fisheye_polynomial_k1");
    bcam->fisheye_polynomial_k2 = RNA_float_get(&ccamera, "fisheye_polynomial_k2");
    bcam->fisheye_polynomial_k3 = RNA_float_get(&ccamera, "fisheye_polynomial_k3");
    bcam->fisheye_polynomial_k4 = RNA_float_get(&ccamera, "fisheye_polynomial_k4");

    bcam->interocular_distance = b_camera.stereo().interocular_distance();
    if (b_camera.stereo().convergence_mode() == BL::CameraStereoData::convergence_mode_PARALLEL) {
      bcam->convergence_distance = FLT_MAX;
    }
    else {
      bcam->convergence_distance = b_camera.stereo().convergence_distance();
    }
    bcam->use_spherical_stereo = b_engine.use_spherical_stereo(b_ob);

    bcam->use_pole_merge = b_camera.stereo().use_pole_merge();
    bcam->pole_merge_angle_from = b_camera.stereo().pole_merge_angle_from();
    bcam->pole_merge_angle_to = b_camera.stereo().pole_merge_angle_to();

    bcam->ortho_scale = b_camera.ortho_scale();

    bcam->lens = b_camera.lens();

    bcam->passepartout_alpha = b_camera.show_passepartout() ? b_camera.passepartout_alpha() : 0.0f;

    if (b_camera.dof().use_dof()) {
      /* allow f/stop number to change aperture_size but still
       * give manual control over aperture radius */
      float fstop = b_camera.dof().aperture_fstop();
      fstop = max(fstop, 1e-5f);

      if (bcam->type == CAMERA_ORTHOGRAPHIC)
        bcam->aperturesize = 1.0f / (2.0f * fstop);
      else
        bcam->aperturesize = (bcam->lens * 1e-3f) / (2.0f * fstop);

      bcam->apertureblades = b_camera.dof().aperture_blades();
      bcam->aperturerotation = b_camera.dof().aperture_rotation();
      bcam->focaldistance = blender_camera_focal_distance(b_engine, b_ob, b_camera, bcam);
      bcam->aperture_ratio = b_camera.dof().aperture_ratio();
    }
    else {
      /* DOF is turned of for the camera. */
      bcam->aperturesize = 0.0f;
      bcam->apertureblades = 0;
      bcam->aperturerotation = 0.0f;
      bcam->focaldistance = 0.0f;
      bcam->aperture_ratio = 1.0f;
    }

    bcam->shift.x = b_engine.camera_shift_x(b_ob, bcam->use_spherical_stereo);
    bcam->shift.y = b_camera.shift_y();

    bcam->sensor_width = b_camera.sensor_width();
    bcam->sensor_height = b_camera.sensor_height();

    if (b_camera.sensor_fit() == BL::Camera::sensor_fit_AUTO)
      bcam->sensor_fit = BlenderCamera::AUTO;
    else if (b_camera.sensor_fit() == BL::Camera::sensor_fit_HORIZONTAL)
      bcam->sensor_fit = BlenderCamera::HORIZONTAL;
    else
      bcam->sensor_fit = BlenderCamera::VERTICAL;
  }
  else if (b_ob_data.is_a(&RNA_Light)) {
    /* Can also look through spot light. */
    BL::SpotLight b_light(b_ob_data);
    float lens = 16.0f / tanf(b_light.spot_size() * 0.5f);
    if (lens > 0.0f) {
      bcam->lens = lens;
    }
  }

  bcam->motion_steps = object_motion_steps(b_ob, b_ob);
}
#endif

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
    /* Set viewplane for panoramic camera. */
    if (viewplane != NULL) {
      *viewplane = bcam->pano_viewplane;

      /* Modify viewplane for camera shift. */
      const float shift_factor = (bcam->pano_aspectratio == 0.0f) ?
                                     1.0f :
                                     *aspectratio / bcam->pano_aspectratio;
      const float dx = bcam->shift.x * shift_factor;
      const float dy = bcam->shift.y * shift_factor;

      viewplane->left += dx;
      viewplane->right += dx;
      viewplane->bottom += dy;
      viewplane->top += dy;
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
      const float dx = 2.0f * (*aspectratio * bcam->shift.x + bcam->offset.x * xaspect * 2.0f);
      const float dy = 2.0f * (*aspectratio * bcam->shift.y + bcam->offset.y * yaspect * 2.0f);

      viewplane->left += dx;
      viewplane->right += dx;
      viewplane->bottom += dy;
      viewplane->top += dy;
    }
  }
}

#ifdef BLENDER_CLIENT
static void blender_camera_sync(
    Camera *cam, BlenderCamera *bcam, int width, int height, const char *viewname
    // PointerRNA *cscene
)
#else
static void blender_camera_sync(Camera *cam,
                                BlenderCamera *bcam,
                                int width,
                                int height,
                                const char *viewname,
                                PointerRNA *cscene)
#endif
{
  float aspectratio, sensor_size;

  /* viewplane */
  BoundBox2D viewplane;
  blender_camera_viewplane(bcam, width, height, &viewplane, &aspectratio, &sensor_size);

  cam->set_viewplane_left(viewplane.left);
  cam->set_viewplane_right(viewplane.right);
  cam->set_viewplane_top(viewplane.top);
  cam->set_viewplane_bottom(viewplane.bottom);

#ifdef BLENDER_CLIENT
  cam->width = bcam->full_width;
  cam->height = bcam->full_height;
#endif

  cam->set_full_width(width);
  cam->set_full_height(height);

  /* panorama sensor */
  if (bcam->type == CAMERA_PANORAMA && (bcam->panorama_type == PANORAMA_FISHEYE_EQUISOLID ||
                                        bcam->panorama_type == PANORAMA_FISHEYE_LENS_POLYNOMIAL)) {
    float fit_xratio = (float)bcam->render_width * bcam->pixelaspect.x;
    float fit_yratio = (float)bcam->render_height * bcam->pixelaspect.y;
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
      cam->set_sensorwidth(sensor_size);
      cam->set_sensorheight(sensor_size * fit_yratio / fit_xratio);
    }
    else {
      cam->set_sensorwidth(sensor_size * fit_xratio / fit_yratio);
      cam->set_sensorheight(sensor_size);
    }
  }

  /* clipping distances */
  cam->set_nearclip(bcam->nearclip);
  cam->set_farclip(bcam->farclip);

  /* type */
  cam->set_camera_type(bcam->type);

  /* panorama */
  cam->set_panorama_type(bcam->panorama_type);
  cam->set_fisheye_fov(bcam->fisheye_fov);
  cam->set_fisheye_lens(bcam->fisheye_lens);
  cam->set_latitude_min(bcam->latitude_min);
  cam->set_latitude_max(bcam->latitude_max);

  cam->set_fisheye_polynomial_k0(bcam->fisheye_polynomial_k0);
  cam->set_fisheye_polynomial_k1(bcam->fisheye_polynomial_k1);
  cam->set_fisheye_polynomial_k2(bcam->fisheye_polynomial_k2);
  cam->set_fisheye_polynomial_k3(bcam->fisheye_polynomial_k3);
  cam->set_fisheye_polynomial_k4(bcam->fisheye_polynomial_k4);

  cam->set_longitude_min(bcam->longitude_min);
  cam->set_longitude_max(bcam->longitude_max);

  /* panorama stereo */
  cam->set_interocular_distance(bcam->interocular_distance);
  cam->set_convergence_distance(bcam->convergence_distance);
  cam->set_use_spherical_stereo(bcam->use_spherical_stereo);

  if (cam->get_use_spherical_stereo()) {
    if (strcmp(viewname, "left") == 0)
      cam->set_stereo_eye(Camera::STEREO_LEFT);
    else if (strcmp(viewname, "right") == 0)
      cam->set_stereo_eye(Camera::STEREO_RIGHT);
    else
      cam->set_stereo_eye(Camera::STEREO_NONE);
  }

  cam->set_use_pole_merge(bcam->use_pole_merge);
  cam->set_pole_merge_angle_from(bcam->pole_merge_angle_from);
  cam->set_pole_merge_angle_to(bcam->pole_merge_angle_to);

  /* anamorphic lens bokeh */
  cam->set_aperture_ratio(bcam->aperture_ratio);

  /* perspective */
#ifdef BLENDER_CLIENT
#  ifdef WITH_VRCLIENT_XTAL
  cam->set_fov(70.0f * (M_PI_F / 180.0f));
#  else
  cam->set_fov(2.0f * atanf((0.5f * sensor_size) / bcam->lens / aspectratio));
#  endif
#else
  cam->set_fov(2.0f * atanf((0.5f * sensor_size) / bcam->lens / aspectratio));
#endif
  cam->set_focaldistance(bcam->focaldistance);
  cam->set_aperturesize(bcam->aperturesize);
  cam->set_blades(bcam->apertureblades);
  cam->set_bladesrotation(bcam->aperturerotation);

  /* transform */
  cam->set_matrix(blender_camera_matrix(bcam->matrix, bcam->type, bcam->panorama_type));

  array<Transform> motion;
  motion.resize(bcam->motion_steps, cam->get_matrix());
  cam->set_motion(motion);
  cam->set_use_perspective_motion(false);

  cam->set_shuttertime(bcam->shuttertime);
  cam->set_fov_pre(cam->get_fov());
  cam->set_fov_post(cam->get_fov());
  cam->set_motion_position(bcam->motion_position);

  cam->set_rolling_shutter_type(bcam->rolling_shutter_type);
  cam->set_rolling_shutter_duration(bcam->rolling_shutter_duration);

#ifndef BLENDER_CLIENT
  cam->set_shutter_curve(bcam->shutter_curve);
#endif

  /* border */
  cam->set_border_left(bcam->border.left);
  cam->set_border_right(bcam->border.right);
  cam->set_border_top(bcam->border.top);
  cam->set_border_bottom(bcam->border.bottom);

  cam->set_viewport_camera_border_left(bcam->viewport_camera_border.left);
  cam->set_viewport_camera_border_right(bcam->viewport_camera_border.right);
  cam->set_viewport_camera_border_top(bcam->viewport_camera_border.top);
  cam->set_viewport_camera_border_bottom(bcam->viewport_camera_border.bottom);

#ifndef BLENDER_CLIENT
  bcam->offscreen_dicing_scale = RNA_float_get(cscene, "offscreen_dicing_scale");
#endif
  cam->set_offscreen_dicing_scale(bcam->offscreen_dicing_scale);

  memcpy(cam->kernel_camera.blender_camera, bcam, sizeof(BlenderCamera));
}

#ifndef BLENDER_CLIENT
/* Sync Render Camera */
void BlenderSync::sync_camera(BL::RenderSettings &b_render,
                              BL::Object &b_override,
                              int width,
                              int height,
                              const char *viewname)
{
  BlenderCamera bcam;
  blender_camera_init(&bcam, b_render);

  /* pixel aspect */
  bcam.pixelaspect.x = b_render.pixel_aspect_x();
  bcam.pixelaspect.y = b_render.pixel_aspect_y();
  bcam.shuttertime = b_render.motion_blur_shutter();

  BL::CurveMapping b_shutter_curve(b_render.motion_blur_shutter_curve());
  curvemapping_to_array(b_shutter_curve, bcam.shutter_curve, RAMP_TABLE_SIZE);

  PointerRNA cscene = RNA_pointer_get(&b_scene.ptr, "cycles");
  bcam.motion_position = (MotionPosition)get_enum(
      cscene, "motion_blur_position", MOTION_NUM_POSITIONS, MOTION_POSITION_CENTER);
  bcam.rolling_shutter_type = (Camera::RollingShutterType)get_enum(
      cscene,
      "rolling_shutter_type",
      Camera::ROLLING_SHUTTER_NUM_TYPES,
      Camera::ROLLING_SHUTTER_NONE);
  bcam.rolling_shutter_duration = RNA_float_get(&cscene, "rolling_shutter_duration");

  /* border */
  if (b_render.use_border()) {
    bcam.border.left = b_render.border_min_x();
    bcam.border.right = b_render.border_max_x();
    bcam.border.bottom = b_render.border_min_y();
    bcam.border.top = b_render.border_max_y();
  }

  /* camera object */
  BL::Object b_ob = b_scene.camera();

  if (b_override)
    b_ob = b_override;

  if (b_ob) {
    BL::Array<float, 16> b_ob_matrix;
    blender_camera_from_object(&bcam, b_engine, b_ob);
    b_engine.camera_model_matrix(b_ob, bcam.use_spherical_stereo, b_ob_matrix);
    bcam.matrix = get_transform(b_ob_matrix);
  }

  /* sync */
  Camera *cam = scene->camera;
  blender_camera_sync(cam, &bcam, width, height, viewname, &cscene);

  /* dicing camera */
  b_ob = BL::Object(RNA_pointer_get(&cscene, "dicing_camera"));
  if (b_ob) {
    BL::Array<float, 16> b_ob_matrix;
    blender_camera_from_object(&bcam, b_engine, b_ob);
    b_engine.camera_model_matrix(b_ob, bcam.use_spherical_stereo, b_ob_matrix);
    bcam.matrix = get_transform(b_ob_matrix);

    blender_camera_sync(scene->dicing_camera, &bcam, width, height, viewname, &cscene);
  }
  else {
    *scene->dicing_camera = *cam;
  }
}

void BlenderSync::sync_camera_motion(
    BL::RenderSettings &b_render, BL::Object &b_ob, int width, int height, float motion_time)
{
  if (!b_ob)
    return;

  Camera *cam = scene->camera;
  BL::Array<float, 16> b_ob_matrix;
  b_engine.camera_model_matrix(b_ob, cam->get_use_spherical_stereo(), b_ob_matrix);
  Transform tfm = get_transform(b_ob_matrix);
  tfm = blender_camera_matrix(tfm, cam->get_camera_type(), cam->get_panorama_type());

  if (motion_time == 0.0f) {
    /* When motion blur is not centered in frame, cam->matrix gets reset. */
    cam->set_matrix(tfm);
  }

  /* Set transform in motion array. */
  int motion_step = cam->motion_step(motion_time);
  if (motion_step >= 0) {
    array<Transform> motion = cam->get_motion();
    motion[motion_step] = tfm;
    cam->set_motion(motion);
  }

  if (cam->get_camera_type() == CAMERA_PERSPECTIVE) {
    BlenderCamera bcam;
    float aspectratio, sensor_size;
    blender_camera_init(&bcam, b_render);

    /* TODO(sergey): Consider making it a part of blender_camera_init(). */
    bcam.pixelaspect.x = b_render.pixel_aspect_x();
    bcam.pixelaspect.y = b_render.pixel_aspect_y();

    blender_camera_from_object(&bcam, b_engine, b_ob);
    blender_camera_viewplane(&bcam, width, height, NULL, &aspectratio, &sensor_size);
    /* TODO(sergey): De-duplicate calculation with camera sync. */
    float fov = 2.0f * atanf((0.5f * sensor_size) / bcam.lens / aspectratio);
    if (fov != cam->get_fov()) {
      VLOG_WORK << "Camera " << b_ob.name() << " FOV change detected.";
      if (motion_time == 0.0f) {
        cam->set_fov(fov);
      }
      else if (motion_time == -1.0f) {
        cam->set_fov_pre(fov);
        cam->set_use_perspective_motion(true);
      }
      else if (motion_time == 1.0f) {
        cam->set_fov_post(fov);
        cam->set_use_perspective_motion(true);
      }
    }
  }
}

/* Sync 3D View Camera */

static void blender_camera_view_subset(BL::RenderEngine &b_engine,
                                       BL::RenderSettings &b_render,
                                       BL::Scene &b_scene,
                                       BL::Object &b_ob,
                                       BL::SpaceView3D &b_v3d,
                                       BL::RegionView3D &b_rv3d,
                                       int width,
                                       int height,
                                       BoundBox2D *view_box,
                                       BoundBox2D *cam_box,
                                       float *view_aspect);

#endif

#ifdef BLENDER_CLIENT
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
  bcam->nearclip = b_v3d.clip_start();
  bcam->farclip = b_v3d.clip_end();
  bcam->lens = b_v3d.lens();
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
#else
static void blender_camera_from_view(BlenderCamera *bcam,
                                     BL::RenderEngine &b_engine,
                                     BL::Scene &b_scene,
                                     BL::SpaceView3D &b_v3d,
                                     BL::RegionView3D &b_rv3d,
                                     int width,
                                     int height,
                                     bool skip_panorama = false)
{
  /* 3d view parameters */
  bcam->nearclip = b_v3d.clip_start();
  bcam->farclip = b_v3d.clip_end();
  bcam->lens = b_v3d.lens();

  bcam->shuttertime = b_scene.render().motion_blur_shutter();

  BL::CurveMapping b_shutter_curve(b_scene.render().motion_blur_shutter_curve());
  curvemapping_to_array(b_shutter_curve, bcam->shutter_curve, RAMP_TABLE_SIZE);

  if (b_rv3d.view_perspective() == BL::RegionView3D::view_perspective_CAMERA) {
    /* camera view */
    BL::Object b_ob = (b_v3d.use_local_camera()) ? b_v3d.camera() : b_scene.camera();

    if (b_ob) {
      blender_camera_from_object(bcam, b_engine, b_ob, skip_panorama);

      if (!skip_panorama && bcam->type == CAMERA_PANORAMA) {
        /* in panorama camera view, we map viewplane to camera border */
        BoundBox2D view_box, cam_box;
        float view_aspect;

        BL::RenderSettings b_render_settings(b_scene.render());
        blender_camera_view_subset(b_engine,
                                   b_render_settings,
                                   b_scene,
                                   b_ob,
                                   b_v3d,
                                   b_rv3d,
                                   width,
                                   height,
                                   &view_box,
                                   &cam_box,
                                   &view_aspect);

        bcam->pano_viewplane = view_box.make_relative_to(cam_box);
        bcam->pano_aspectratio = view_aspect;
      }
      else {
        /* magic zoom formula */
        bcam->zoom = (float)b_rv3d.view_camera_zoom();
        bcam->zoom = (1.41421f + bcam->zoom / 50.0f);
        bcam->zoom *= bcam->zoom;
        bcam->zoom = 2.0f / bcam->zoom;

        /* offset */
        bcam->offset = get_float2(b_rv3d.view_camera_offset());
      }
    }
  }
  else if (b_rv3d.view_perspective() == BL::RegionView3D::view_perspective_ORTHO) {
    /* orthographic view */
    bcam->farclip *= 0.5f;
    bcam->nearclip = -bcam->farclip;

    float sensor_size;
    if (bcam->sensor_fit == BlenderCamera::VERTICAL)
      sensor_size = bcam->sensor_height;
    else
      sensor_size = bcam->sensor_width;

    bcam->type = CAMERA_ORTHOGRAPHIC;
    bcam->ortho_scale = b_rv3d.view_distance() * sensor_size / b_v3d.lens();
  }

  bcam->zoom *= 2.0f;

  /* 3d view transform */
  bcam->matrix = transform_inverse(get_transform(b_rv3d.view_matrix()));

  /* dimensions */
  bcam->full_width = width;
  bcam->full_height = height;
}

static void blender_camera_view_subset(BL::RenderEngine &b_engine,
                                       BL::RenderSettings &b_render,
                                       BL::Scene &b_scene,
                                       BL::Object &b_ob,
                                       BL::SpaceView3D &b_v3d,
                                       BL::RegionView3D &b_rv3d,
                                       int width,
                                       int height,
                                       BoundBox2D *view_box,
                                       BoundBox2D *cam_box,
                                       float *view_aspect)
{
  BoundBox2D cam, view;
  float cam_aspect, sensor_size;

  /* Get viewport viewplane. */
  BlenderCamera view_bcam;
  blender_camera_init(&view_bcam, b_render);
  blender_camera_from_view(&view_bcam, b_engine, b_scene, b_v3d, b_rv3d, width, height, true);

  blender_camera_viewplane(&view_bcam, width, height, &view, view_aspect, &sensor_size);

  /* Get camera viewplane. */
  BlenderCamera cam_bcam;
  blender_camera_init(&cam_bcam, b_render);
  blender_camera_from_object(&cam_bcam, b_engine, b_ob, true);

  /* Camera border is affect by aspect, viewport is not. */
  cam_bcam.pixelaspect.x = b_render.pixel_aspect_x();
  cam_bcam.pixelaspect.y = b_render.pixel_aspect_y();

  blender_camera_viewplane(
      &cam_bcam, cam_bcam.full_width, cam_bcam.full_height, &cam, &cam_aspect, &sensor_size);

  /* Return */
  *view_box = view * (1.0f / *view_aspect);
  *cam_box = cam * (1.0f / cam_aspect);
}

static void blender_camera_border_subset(BL::RenderEngine &b_engine,
                                         BL::RenderSettings &b_render,
                                         BL::Scene &b_scene,
                                         BL::SpaceView3D &b_v3d,
                                         BL::RegionView3D &b_rv3d,
                                         BL::Object &b_ob,
                                         int width,
                                         int height,
                                         const BoundBox2D &border,
                                         BoundBox2D *result)
{
  /* Determine camera viewport subset. */
  BoundBox2D view_box, cam_box;
  float view_aspect;
  blender_camera_view_subset(b_engine,
                             b_render,
                             b_scene,
                             b_ob,
                             b_v3d,
                             b_rv3d,
                             width,
                             height,
                             &view_box,
                             &cam_box,
                             &view_aspect);

  /* Determine viewport subset matching given border. */
  cam_box = cam_box.make_relative_to(view_box);
  *result = cam_box.subset(border);
}

static void blender_camera_border(BlenderCamera *bcam,
                                  BL::RenderEngine &b_engine,
                                  BL::RenderSettings &b_render,
                                  BL::Scene &b_scene,
                                  BL::SpaceView3D &b_v3d,
                                  BL::RegionView3D &b_rv3d,
                                  int width,
                                  int height)
{
  bool is_camera_view;

  /* camera view? */
  is_camera_view = b_rv3d.view_perspective() == BL::RegionView3D::view_perspective_CAMERA;

  if (!is_camera_view) {
    /* for non-camera view check whether render border is enabled for viewport
     * and if so use border from 3d viewport
     * assume viewport has got correctly clamped border already
     */
    if (b_v3d.use_render_border()) {
      bcam->border.left = b_v3d.render_border_min_x();
      bcam->border.right = b_v3d.render_border_max_x();
      bcam->border.bottom = b_v3d.render_border_min_y();
      bcam->border.top = b_v3d.render_border_max_y();
    }
    return;
  }

  BL::Object b_ob = (b_v3d.use_local_camera()) ? b_v3d.camera() : b_scene.camera();

  if (!b_ob)
    return;

  /* Determine camera border inside the viewport. */
  BoundBox2D full_border;
  blender_camera_border_subset(b_engine,
                               b_render,
                               b_scene,
                               b_v3d,
                               b_rv3d,
                               b_ob,
                               width,
                               height,
                               full_border,
                               &bcam->viewport_camera_border);

  if (b_render.use_border()) {
    bcam->border.left = b_render.border_min_x();
    bcam->border.right = b_render.border_max_x();
    bcam->border.bottom = b_render.border_min_y();
    bcam->border.top = b_render.border_max_y();
  }
  else if (bcam->passepartout_alpha == 1.0f) {
    bcam->border = full_border;
  }
  else {
    return;
  }

  /* Determine viewport subset matching camera border. */
  blender_camera_border_subset(b_engine,
                               b_render,
                               b_scene,
                               b_v3d,
                               b_rv3d,
                               b_ob,
                               width,
                               height,
                               bcam->border,
                               &bcam->border);
  bcam->border = bcam->border.clamp();
}

void BlenderSync::sync_view(BL::SpaceView3D &b_v3d,
                            BL::RegionView3D &b_rv3d,
                            int width,
                            int height)
{
  BlenderCamera bcam;
  BL::RenderSettings b_render_settings(b_scene.render());
  blender_camera_init(&bcam, b_render_settings);
  blender_camera_from_view(&bcam, b_engine, b_scene, b_v3d, b_rv3d, width, height);
  blender_camera_border(&bcam, b_engine, b_render_settings, b_scene, b_v3d, b_rv3d, width, height);
  PointerRNA cscene = RNA_pointer_get(&b_scene.ptr, "cycles");
  blender_camera_sync(scene->camera, &bcam, width, height, "", &cscene);

  /* dicing camera */
  BL::Object b_ob = BL::Object(RNA_pointer_get(&cscene, "dicing_camera"));
  if (b_ob) {
    BL::Array<float, 16> b_ob_matrix;
    blender_camera_from_object(&bcam, b_engine, b_ob);
    b_engine.camera_model_matrix(b_ob, bcam.use_spherical_stereo, b_ob_matrix);
    bcam.matrix = get_transform(b_ob_matrix);

    blender_camera_sync(scene->dicing_camera, &bcam, width, height, "", &cscene);
  }
  else {
    *scene->dicing_camera = *scene->camera;
  }
}

BufferParams BlenderSync::get_buffer_params(
    BL::SpaceView3D &b_v3d, BL::RegionView3D &b_rv3d, Camera *cam, int width, int height)
{
  BufferParams params;
  bool use_border = false;

  params.full_width = width;
  params.full_height = height;

  if (b_v3d && b_rv3d && b_rv3d.view_perspective() != BL::RegionView3D::view_perspective_CAMERA)
    use_border = b_v3d.use_render_border();
  else
    /* the camera can always have a passepartout */
    use_border = true;

  if (use_border) {
    /* border render */
    /* the viewport may offset the border outside the view */
    BoundBox2D border = cam->border.clamp();
    params.full_x = (int)(border.left * (float)width);
    params.full_y = (int)(border.bottom * (float)height);
    params.width = (int)(border.right * (float)width) - params.full_x;
    params.height = (int)(border.top * (float)height) - params.full_y;

    /* survive in case border goes out of view or becomes too small */
    params.width = max(params.width, 1);
    params.height = max(params.height, 1);
  }
  else {
    params.width = width;
    params.height = height;
  }

  params.window_width = params.width;
  params.window_height = params.height;

  return params;
}

#endif

#ifdef BLENDER_CLIENT

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
  // blender_camera_border(
  //    &bcam, /*b_engine, b_render_settings, b_scene,*/ b_v3d, b_rv3d, width, height);
  // PointerRNA cscene = RNA_pointer_get(&b_scene.ptr, "cycles");
  blender_camera_sync(scene_camera, &bcam, width, height, "" /*, &cscene*/);
  scene_camera->update();

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
Camera g_scene_camera_right;

#endif

CCL_NAMESPACE_END

#ifdef BLENDER_CLIENT

namespace cyclesphi {
namespace kernel {

void view_to_kernel_camera(char *kdata, cyclesphi_data *cd)
{
  ccl::SpaceView3D b_v3d;
  b_v3d._clip_start = cd->cam.clip_start;
  b_v3d._clip_end = cd->cam.clip_end;
  b_v3d._lens = cd->cam.lens;

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

  // b_rv3d.transform_inverse_view_matrix = ccl::transform_identity();

  b_rv3d.view_camera_zoom = cd->cam.view_camera_zoom;
  b_rv3d.view_camera_offset[0] = cd->cam.view_camera_offset[0];
  b_rv3d.view_camera_offset[1] = cd->cam.view_camera_offset[1];

  // memcpy((char*)&ccl::g_scene_camera.kernel_camera, kcam, sizeof(ccl::KernelCamera));

  ccl::sync_view(&ccl::g_scene_camera,
                 (cd->cam.use_view_camera) ? &b_obj : NULL,
                 b_v3d,
                 b_rv3d,
                 cd->width,
                 cd->height);

  memcpy(&((ccl::KernelData *)kdata)->cam,
         (char *)&ccl::g_scene_camera.kernel_camera,
         sizeof(ccl::KernelCamera));
}

void bcam_to_kernel_camera(char *kdata, char *bc, int width, int height)
{
  ccl::BlenderCamera *bcam = (ccl::BlenderCamera *)((ccl::KernelData *)kdata)->cam.blender_camera;
  ccl::blender_camera_sync(&ccl::g_scene_camera, bcam, width, height, "");
  ccl::g_scene_camera.update();

  memcpy(&((ccl::KernelData *)kdata)->cam,
         (char *)&ccl::g_scene_camera.kernel_camera,
         sizeof(ccl::KernelCamera));
}

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_CLIENT_ULTRAGRID)
void view_to_kernel_camera_right(char *kdata, cyclesphi_data *cd)
{
  ccl::SpaceView3D b_v3d;
  b_v3d._clip_start = cd->cam_right.clip_start;
  b_v3d._clip_end = cd->cam_right.clip_end;
  b_v3d._lens = cd->cam_right.lens;

  ccl::ObjectCamera b_obj;
  b_obj.clip_start = cd->cam_right.clip_start;
  b_obj.clip_end = cd->cam_right.clip_end;
  b_obj.lens = cd->cam_right.lens;
  b_obj.sensor_fit = cd->cam_right.sensor_fit;
  b_obj.sensor_width = cd->cam_right.sensor_width;
  b_obj.sensor_height = cd->cam_right.sensor_height;
  b_obj.shift_x = cd->cam_right.shift_x;
  b_obj.shift_y = cd->cam_right.shift_y;

  // b_obj.stereo_eye = STEREO_RIGHT;
  //  b_obj.interocular_distance = cd->cam_right.interocular_distance;
  //  b_obj.convergence_distance = cd->cam_right.convergence_distance;

  ccl::RegionView3D b_rv3d;
  memcpy(&b_rv3d.transform_inverse_view_matrix,
         cd->cam_right.transform_inverse_view_matrix,
         sizeof(ccl::Transform));

  b_rv3d.view_camera_zoom = cd->cam_right.view_camera_zoom;
  b_rv3d.view_camera_offset[0] = cd->cam_right.view_camera_offset[0];
  b_rv3d.view_camera_offset[1] = cd->cam_right.view_camera_offset[1];

  ccl::sync_view(&ccl::g_scene_camera_right,
                 (cd->cam_right.use_view_camera) ? &b_obj : NULL,
                 b_v3d,
                 b_rv3d,
                 cd->width,
                 cd->height);

  memcpy(&((ccl::KernelData *)kdata)->cam,
         (char *)&ccl::g_scene_camera_right.kernel_camera,
         sizeof(ccl::KernelCamera));
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

  mul_m444((float *)&bcam->matrix, right_eye, (float *)&bcam->matrix);

  // bcam->convergence_distance = FLT_MAX;
  // bcam->interocular_distance = 0.065f;
  // bcam->use_spherical_stereo = true;

  ccl::blender_camera_sync(&ccl::g_scene_camera_right, bcam, width, height, "");
  ccl::g_scene_camera_right.update();

  memcpy(&((ccl::KernelData *)kdata)->cam,
         (char *)&ccl::g_scene_camera_right.kernel_camera,
         sizeof(ccl::KernelCamera));
}

#  endif

}  // namespace kernel
}  // namespace cyclesphi

#endif
