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

#ifndef __CLIENT_API_H__
#define __CLIENT_API_H__

//#include <cstdio>
#include <stdint.h>

/////////////////////////cycles//////////////////////////////////////
#define CLIENT_TAG_CYCLES_start 1000

#define CLIENT_TAG_CYCLES_const_copy 1001
#define CLIENT_TAG_CYCLES_tex_copy 1002
#define CLIENT_TAG_CYCLES_path_trace 1003
//#define CLIENT_TAG_CYCLES_branched_path_trace   1004
//#define CLIENT_TAG_CYCLES_film_convert_half   1005
//#define CLIENT_TAG_CYCLES_film_convert_byte   1006
//#define CLIENT_TAG_CYCLES_bake      1007
//#define CLIENT_TAG_CYCLES_shader      1008

#define CLIENT_TAG_CYCLES_alloc_kg 1010
#define CLIENT_TAG_CYCLES_free_kg 1011
#define CLIENT_TAG_CYCLES_mem_alloc 1012
#define CLIENT_TAG_CYCLES_mem_copy_to 1013
//#define CLIENT_TAG_CYCLES_mem_copy_from    1014
#define CLIENT_TAG_CYCLES_mem_zero 1015
#define CLIENT_TAG_CYCLES_mem_free 1016

//#define CLIENT_TAG_CYCLES_path_trace_buffer   1017
//#define CLIENT_TAG_CYCLES_path_trace_rng_state  1018
//#define CLIENT_TAG_CYCLES_path_trace_buffer_sample 1019
//#define CLIENT_TAG_CYCLES_path_trace_rgba             1020
//#define CLIENT_TAG_CYCLES_path_trace_rng              1021
//#define CLIENT_TAG_CYCLES_tex_copy_data    1022
//#define CLIENT_TAG_CYCLES_mem_copy_to_data   1023
//#define CLIENT_TAG_CYCLES_mem_copy_from_data   1024
#define CLIENT_TAG_CYCLES_tex_free 1025
#define CLIENT_TAG_CYCLES_load_textures 1026
#define CLIENT_TAG_CYCLES_close_connection 1027

#define CLIENT_TAG_CYCLES_tex_interp 1030
#define CLIENT_TAG_CYCLES_tex_interp3d 1031
#define CLIENT_TAG_CYCLES_path_trace_finish 1032

#define CLIENT_TAG_CYCLES_tex_info 1033

#define CLIENT_TAG_CYCLES_denoising_non_local_means 1050
#define CLIENT_TAG_CYCLES_denoising_construct_transform 1051
#define CLIENT_TAG_CYCLES_denoising_reconstruct 1052
#define CLIENT_TAG_CYCLES_denoising_combine_halves 1053
#define CLIENT_TAG_CYCLES_denoising_divide_shadow 1054
#define CLIENT_TAG_CYCLES_denoising_get_feature 1055
#define CLIENT_TAG_CYCLES_denoising_detect_outliers 1056

#define CLIENT_TAG_CYCLES_mem_alloc_sub_ptr 1057
#define CLIENT_TAG_CYCLES_receive_render_buffer 1058

#define CLIENT_TAG_CYCLES_path_to_cache 1059

#define CLIENT_TAG_CYCLES_build_bvh 1060
//#define CLIENT_TAG_CYCLES_blender_camera 1061

#define CLIENT_TAG_CYCLES_frame_info 1062

#define CLIENT_TAG_CYCLES_end 1999

//////////////////////////////OTHER//////////////////////////////////////
#define MPI_NAME_MAX_LENGTH 256
#define DEVICE_PTR unsigned long long
#define CLIENT_SIZE_T unsigned long long

//#ifdef WITH_CLIENT_MPI

#  define SIZE_UCHAR4 (sizeof(unsigned short) * 4)

//#else
//
//#  ifdef WITH_SIZE_UCHAR3
//#    define SIZE_UCHAR4 (sizeof(unsigned char) * 3)
//#  else
//#    define SIZE_UCHAR4 (sizeof(unsigned char) * 4)
//#  endif
//
//#endif

#define SIZE_FLOAT4 (sizeof(float) * 4)

#define CLIENT_SWAP_BUFFERS 1

//#define CLIENT_MPI_LOAD_BALANCING_LINES
//#define CLIENT_MPI_LOAD_BALANCING_SAMPLES
//#define MPI_CACHE_FILE2
//#define MPI_CACHE_FILE3
//#define MPI_REMOTE_TEX

//#define MPI_ANIMATION

#define CLIENT_REQUEST_NONE 0
#define CLIENT_REQUEST_FINISH 1
#define CLIENT_REQUEST_IMAGE 2

#define client_static_assert_align(st, \
                                   align)  // static_assert((sizeof(st) % (align) == 0),
                                           // "Structure must be strictly aligned")  // NOLINT

#define CLIENT_TRACER  // printf("%s\n", __FUNCTION__);fflush(0);

//#define CLIENT_DEBUG_PRINTF
#ifdef CLIENT_DEBUG_PRINTF
#  define CLIENT_DEBUG_PRINTF4(t1, t2, t3, t4) printf(t1, t2, t3, t4)
#  define CLIENT_DEBUG_PRINTF3(t1, t2, t3) printf(t1, t2, t3)
#  define CLIENT_DEBUG_PRINTF2(t1, t2) printf(t1, t2)
#  define CLIENT_DEBUG_PRINTF1(t1) printf(t1)
#else
#  define CLIENT_DEBUG_PRINTF4(t1, t2, t3, t4)
#  define CLIENT_DEBUG_PRINTF3(t1, t2, t3)
#  define CLIENT_DEBUG_PRINTF2(t1, t2)
#  define CLIENT_DEBUG_PRINTF1(t1)
#endif

#ifdef WITH_CLIENT_MPI_VRCLIENT
#  include "cyclesphi_data.h"
#endif

#define CLIENT_DEVICE_ID 0

// namespace cyclesphi {
// namespace kernel {
/////////////////////////////////CYCLES///////////////////////////////////////////////
typedef struct client_build_bvh_struct {
  int operation;
  DEVICE_PTR build_input;
  CLIENT_SIZE_T build_size;
  int num_motion_steps;

  int pad[2];
} client_build_bvh_struct;
client_static_assert_align(client_build_bvh_struct, 16);

// typedef struct client_blender_camera_struct {
//  DEVICE_PTR mem;
//  CLIENT_SIZE_T size;
//} client_blender_camera_struct;
// client_static_assert_align(client_blender_camera_struct, 16);

typedef struct client_buffer_passes {
  char name[MPI_NAME_MAX_LENGTH];

  int type;
  int mode;
  int include_albedo;
  int offset;

} client_buffer_passes;
client_static_assert_align(client_buffer_passes, 16);

typedef struct client_tex_interp {
  int id;
  float x;
  float y;
  float z;
  int type;

  int pad[3];
} client_tex_interp;
client_static_assert_align(client_tex_interp, 16);

/* Path Tracing */
typedef struct client_path_trace_struct {
  DEVICE_PTR buffer;
  DEVICE_PTR pixels;
  int start_sample;
  int num_samples;
  int sample_offset;
  int tile_x;
  int tile_y;
  int offset;
  int stride;
  int tile_h;
  int tile_w;
  int tile_h2;
  int tile_w2;
  int pass_stride;
  bool use_load_balancing;
  int tile_step;
  int compress;

  int has_shadow_catcher;
  int max_shaders;
  unsigned int kernel_features;
  unsigned int volume_stack_size;

#ifdef WITH_CLIENT_MPI_VRCLIENT
  cyclesphi_data cdata;
  int pad[2];
#endif

} client_path_trace_struct;
client_static_assert_align(client_path_trace_struct, 16);

typedef struct client_load_textures_struct {
  CLIENT_SIZE_T texture_info_size;
  int pad[2];
} client_load_textures_struct;
client_static_assert_align(client_load_textures_struct, 16);

/* Film */
typedef struct client_film_convert_struct {
  DEVICE_PTR rgba_byte;
  DEVICE_PTR rgba_half;
  DEVICE_PTR buffer;
  float sample_scale;
  int offset;
  int stride;
  int task_x;
  int task_y;
  int task_h;
  int task_w;
  int pad[3];
} client_film_convert_struct;
client_static_assert_align(client_film_convert_struct, 16);

/* Shader Evaluation */
/*typedef struct client_bake_struct {
    DEVICE_PTR input;
    DEVICE_PTR output;
    int type;
    int task_shader_x;
    int task_shader_w;
    int offset;
    int sample;
} client_bake_struct;*/

typedef struct client_shader_struct {
  DEVICE_PTR input;
  DEVICE_PTR output;
  int type;
  int task_shader_x;
  int task_shader_w;
  int sample;

} client_shader_struct;
client_static_assert_align(client_shader_struct, 16);

typedef struct client_mem_struct {
  char name[MPI_NAME_MAX_LENGTH];
  DEVICE_PTR mem;
  CLIENT_SIZE_T offset;
  CLIENT_SIZE_T memSize;
  DEVICE_PTR mem_sub;

} client_mem_struct;
client_static_assert_align(client_mem_struct, 16);

typedef struct client_const_copy_struct {
  char name[MPI_NAME_MAX_LENGTH];
  DEVICE_PTR host;
  CLIENT_SIZE_T size;
  bool read_data;

  int pad[3];
} client_const_copy_struct;
client_static_assert_align(client_const_copy_struct, 16);

typedef struct client_tex_copy_struct {
  char name[MPI_NAME_MAX_LENGTH];
  DEVICE_PTR mem;
  CLIENT_SIZE_T data_size;
  CLIENT_SIZE_T mem_size;

  int pad[2];
} client_tex_copy_struct;
client_static_assert_align(client_tex_copy_struct, 16);

typedef struct client_tex_info_struct {
  char name[MPI_NAME_MAX_LENGTH];

  DEVICE_PTR mem;
  CLIENT_SIZE_T size;

  CLIENT_SIZE_T data_width;
  CLIENT_SIZE_T data_height;

  CLIENT_SIZE_T data_depth;
  int data_type;
  int data_elements;

  int interpolation;
  int extension;

  int pad[2];
} client_tex_info_struct;
client_static_assert_align(client_tex_info_struct, 16);


typedef struct client_frame_info_struct {
  int current_frame;
  int current_frame_preview;
  int caching_enabled;

  int pad[1];
} client_frame_info_struct;
client_static_assert_align(client_frame_info_struct, 16);


// typedef struct client_denoising_task_struct {
//    /* Parameters of the denoising algorithm. */
//    int radius;
//    float nlm_k_2;
//    float pca_threshold;
//
//    /* Parameters of the RenderBuffers. */
//    int render_buffer_offset;
//    int render_buffer_pass_stride;
//    int render_buffer_samples;
//
//    /* Pointer and parameters of the target buffer. */
//    int target_buffer_offset;
//    int target_buffer_stride;
//    int target_buffer_pass_stride;
//    int target_buffer_denoising_clean_offset;
//    DEVICE_PTR target_buffer_ptr;
//
//    //TileInfo *tile_info;
//    int tile_info_offsets[9];
//    int tile_info_strides[9];
//    int tile_info_x[4];
//    int tile_info_y[4];
//    long long int tile_info_buffers[9];
//
//    DEVICE_PTR tile_info_mem_device_pointer;
//
//    int rect_x, rect_y, rect_z, rect_w;
//    int filter_area_x, filter_area_y, filter_area_z, filter_area_w;
//
//    /* Stores state of the current Reconstruction operation,
//     * which is accessed by the device in order to perform the operation. */
//    int reconstruction_state_filter_window_x, reconstruction_state_filter_window_y,
//    reconstruction_state_filter_window_z, reconstruction_state_filter_window_w; int
//    reconstruction_state_buffer_params_x, reconstruction_state_buffer_params_y,
//    reconstruction_state_buffer_params_z, reconstruction_state_buffer_params_w;
//
//    int reconstruction_state_source_w;
//    int reconstruction_state_source_h;
//
//    /* Stores state of the current NLM operation,
//     * which is accessed by the device in order to perform the operation. */
//    int nlm_state_r;      /* Search radius of the filter. */
//    int nlm_state_f;      /* Patch size of the filter. */
//    float nlm_state_a;    /* Variance compensation factor in the MSE estimation. */
//    float nlm_state_k_2;  /* Squared value of the k parameter of the filter. */
//
//    DEVICE_PTR storage_transform_device_pointer;
//    DEVICE_PTR storage_rank_device_pointer;
//
//    DEVICE_PTR storage_XtWX_device_pointer;
//    CLIENT_SIZE_T storage_XtWX_size;
//
//    DEVICE_PTR storage_XtWY_device_pointer;
//    CLIENT_SIZE_T storage_XtWY_size;
//
//    int storage_w;
//    int storage_h;
//
//    int buffer_pass_stride;
//    int buffer_passes;
//    int buffer_stride;
//    int buffer_h;
//    int buffer_width;
//    DEVICE_PTR buffer_mem_device_pointer;
//    DEVICE_PTR buffer_temporary_mem_device_pointer;
//
//    bool buffer_gpu_temporary_mem;
//
//    int pad[2];
//
//} client_denoising_task_struct;
// client_static_assert_align(client_denoising_task_struct, 16);

///////////////////////////////////////////////////////////////////////
// typedef struct client_denoising_non_local_means_struct {
//    DEVICE_PTR image_ptr;
//    DEVICE_PTR guide_ptr;
//    DEVICE_PTR variance_ptr;
//    DEVICE_PTR out_ptr;
//
//}client_denoising_non_local_means_struct;
// client_static_assert_align(client_denoising_non_local_means_struct, 16);
//
// typedef struct client_denoising_construct_transform_struct {
//    int pad[4];
//}client_denoising_construct_transform_struct;
// client_static_assert_align(client_denoising_construct_transform_struct, 16);
//
// typedef struct client_denoising_reconstruct_struct {
//    DEVICE_PTR color_ptr;
//    DEVICE_PTR color_variance_ptr;
//    DEVICE_PTR output_ptr;
//    int pad[2];
//}client_denoising_reconstruct_struct;
// client_static_assert_align(client_denoising_reconstruct_struct, 16);
//
// typedef struct client_denoising_combine_halves_struct {
//    DEVICE_PTR a_ptr;
//    DEVICE_PTR b_ptr;
//    DEVICE_PTR mean_ptr;
//    DEVICE_PTR variance_ptr;
//    int r;
//    int rect[4];
//
//    int pad[3];
//}client_denoising_combine_halves_struct;
// client_static_assert_align(client_denoising_combine_halves_struct, 16);
//
// typedef struct client_denoising_divide_shadow_struct {
//    DEVICE_PTR a_ptr;
//    DEVICE_PTR b_ptr;
//    DEVICE_PTR sample_variance_ptr;
//    DEVICE_PTR sv_variance_ptr;
//    DEVICE_PTR buffer_variance_ptr;
//    int pad[2];
//}client_denoising_divide_shadow_struct;
// client_static_assert_align(client_denoising_divide_shadow_struct, 16);
//
// typedef struct client_denoising_get_feature_struct {
//    int mean_offset;
//    int variance_offset;
//    DEVICE_PTR mean_ptr;
//    DEVICE_PTR variance_ptr;
//    int pad[2];
//}client_denoising_get_feature_struct;
// client_static_assert_align(client_denoising_get_feature_struct, 16);
//
// typedef struct client_denoising_detect_outliers_struct {
//    DEVICE_PTR image_ptr;
//    DEVICE_PTR variance_ptr;
//    DEVICE_PTR depth_ptr;
//    DEVICE_PTR output_ptr;
//}client_denoising_detect_outliers_struct;
// client_static_assert_align(client_denoising_detect_outliers_struct, 16);

///////////////////////////////////////////////////////////////////////
typedef struct client_kernel_struct {
  int client_tag;
  int world_size;
  int world_rank;
  char *comm_data;

  ////////////cycles///////////////////////
  client_path_trace_struct client_path_trace_data;
  client_film_convert_struct client_film_convert_data;
  /*client_bake_struct client_bake_data;*/
  client_shader_struct client_shader_data;
  client_mem_struct client_mem_data;
  client_const_copy_struct client_const_copy_data;
  client_tex_copy_struct client_tex_copy_data;
  client_load_textures_struct client_load_textures_data;
  client_tex_info_struct client_tex_info_data;
  // client_denoising_task_struct client_denoising_task_data;
  client_build_bvh_struct client_build_bvh_data;
  // client_blender_camera_struct client_blender_camera_data;
  client_frame_info_struct client_frame_info_data;
  //////////////////////////////////////////
  // client_denoising_non_local_means_struct client_denoising_non_local_means_data;
  // client_denoising_construct_transform_struct client_denoising_construct_transform_data;
  // client_denoising_reconstruct_struct client_denoising_reconstruct_data;
  // client_denoising_combine_halves_struct client_denoising_combine_halves_data;
  // client_denoising_divide_shadow_struct client_denoising_divide_shadow_data;
  // client_denoising_get_feature_struct client_denoising_get_feature_data;
  // client_denoising_detect_outliers_struct client_denoising_detect_outliers_data;

  // int pad[2];

} client_kernel_struct;
client_static_assert_align(client_kernel_struct, 16);

//}}

#endif /* __CLIENT_API_H__ */
