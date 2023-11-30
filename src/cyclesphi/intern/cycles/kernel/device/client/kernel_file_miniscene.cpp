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

#include "kernel_file_miniscene.h"

#include "../../../util/types.h"
#include <cstdio>
#include <omp.h>
#include <string.h>
#include <vector>

//#include "device/client/device_impl.h"
#include "kernel/device/cpu/compat.h"
#include "kernel/device/cpu/globals.h"
//#include "kernel/device/cpu/image.h"
//
//#include "kernel/util/differential.h"
//#include "kernel/util/color.h"
//#include "kernel/geom/geom.h"
//
//#include "kernel/integrator/state.h"
//#include "kernel/integrator/state_flow.h"
//#include "kernel/integrator/shader_eval.h"
//
//#include "kernel/closure/bsdf_microfacet.h"
//
//#include "kernel/svm/svm.h"

//#include "kernel/device/cpu/kernel.h"
//#define KERNEL_ARCH cpu_mini
//#include "kernel/device/cpu/kernel_arch_impl.h"

#include "miniScene/Scene.h"

namespace cyclesphi {
namespace kernel {
namespace miniscene {

bool init_kernel_global();
void get_kernel_data(client_kernel_struct *data, int client_tag);

void close_kernelglobal()
{
}

bool init_kernel_global()
{
  return true;
}

void write_data_kernelglobal(void *data, size_t size)
{
}

bool read_data_kernelglobal(void *data, size_t size)
{
  return true;
}

/////////////////////////////////////////////////////

void get_kernel_data(client_kernel_struct *data, int client_tag)
{
}

bool delete_from_cache(client_kernel_struct &data)
{
  return true;
}

void send_kernel_data()
{
}

void send_to_cache(client_kernel_struct &data, void *mem = NULL, size_t size = 0)
{
}

int count_devices()
{
  return 1;
}

bool read_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size)
{
  return true;
}

bool read_cycles_data(const char *filename, char *buffer, size_t offset, size_t size)
{
  return true;
}

bool write_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size)
{
  return true;
}

bool write_cycles_data(const char *filename, char *buffer, size_t offset, size_t size)
{
  return true;
}

////////////////////////////////////////////////////////////////////////////

bool is_preprocessing()
{
  return false;
}

bool is_postprocessing()
{
  return false;
}

int get_additional_samples()
{
  return 0;
}

void receive_render_buffer(char *buffer_pixels, int tile_h, int tile_w, int pass_stride)
{
}

void receive_path_trace(int offset,
                        int stride,
                        int tile_x,
                        int tile_y,
                        int tile_h,
                        int tile_w,
                        int num_samples,
                        size_t pass_stride_sizeof,  // bool compress,
                        char *buffer_pixels,
                        char *task_bin,
                        char *tile_bin,
                        char *kg_bin,
                        void (*tex_update)(bool, char *, int, float, float, float, int, float *),
                        void (*update_progress)(char *, char *, int, int))
{
}

void const_copy(const char *name,
                char *host_bin,
                size_t size,
                DEVICE_PTR host_ptr,
                client_kernel_struct *_client_data)
{
}

void load_textures(size_t size, client_kernel_struct *_client_data)
{
}

void tex_copy(const char *name,
              void *mem,
              size_t data_size,
              size_t mem_size,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{
}

void alloc_kg(client_kernel_struct *_client_data)
{
}

void free_kg(client_kernel_struct *_client_data)
{
}

void mem_alloc(const char *name,
               DEVICE_PTR mem,
               size_t memSize,
               DEVICE_PTR host_ptr,
               client_kernel_struct *_client_data)
{
}

void mem_alloc_sub_ptr(const char *name,
                       DEVICE_PTR mem,
                       size_t offset,
                       DEVICE_PTR mem_sub,
                       client_kernel_struct *_client_data)
{
}

void mem_copy_to(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 size_t offset,
                 DEVICE_PTR host_ptr,
                 client_kernel_struct *_client_data)
{
}

void mem_zero(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              size_t offset,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{
}

void mem_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
{
}

void tex_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_client_data)
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
}

void build_optix_bvh(int operation, char *build_input, size_t build_size, int num_motion_steps)
{
}

void convert_rgb_to_half(unsigned short *destination, uchar *source, int tile_h, int tile_w)
{
}

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
}
/////////////////////////////////////////////////////////////////////

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
                int pass_stride,
                int has_shadow_catcher,
                int max_shaders,
                unsigned int kernel_features,
                unsigned int volume_stack_size,
                DEVICE_PTR buffer_host_ptr,
                DEVICE_PTR pixels_host_ptr)
{
}
/////////////////////////////////////////////////////////////////////
// ccl::float3 fix_xyz(ccl::float3 xyz_)
//{
//  ccl::float3 xyz;
//  xyz.x = xyz_.x;
//  xyz.y = xyz_.y;
//  xyz.z = xyz_.z;
//
//  xyz.w = xyz_.w;
//
//  return xyz;
//}

// void transform_decompose(ccl::DecomposedTransform *decomp, const ccl::Transform *tfm)
//{
//  /* extract translation */
//  decomp->y = ccl::make_float4(tfm->x.w, tfm->y.w, tfm->z.w, 0.0f);
//
//  /* extract rotation */
//  ccl::Transform M = *tfm;
//  M.x.w = 0.0f;
//  M.y.w = 0.0f;
//  M.z.w = 0.0f;
//
//  ccl::float3 colx = ccl::transform_get_column(&M, 0);
//  ccl::float3 coly = ccl::transform_get_column(&M, 1);
//  ccl::float3 colz = ccl::transform_get_column(&M, 2);
//
//  /* extract scale and shear first */
//  ccl::float3 scale, shear;
//  scale.x = ccl::len(colx);
//  colx = ccl::safe_divide(colx, scale.x);
//  shear.z = ccl::dot(colx, coly);
//  coly -= shear.z * colx;
//  scale.y = ccl::len(coly);
//  coly = ccl::safe_divide(coly, scale.y);
//  shear.y = ccl::dot(colx, colz);
//  colz -= shear.y * colx;
//  shear.x = ccl::dot(coly, colz);
//  colz -= shear.x * coly;
//  scale.z = ccl::len(colz);
//  colz = ccl::safe_divide(colz, scale.z);
//
//  ccl::transform_set_column(&M, 0, colx);
//  ccl::transform_set_column(&M, 1, coly);
//  ccl::transform_set_column(&M, 2, colz);
//
//  if (ccl::transform_negative_scale(M)) {
//    scale *= -1.0f;
//    M = M * ccl::transform_scale(-1.0f, -1.0f, -1.0f);
//  }
//
//  decomp->x = ccl::transform_to_quat(M);
//
//  decomp->y.w = scale.x;
//  decomp->z = ccl::make_float4(shear.z, shear.y, 0.0f, scale.y);
//  decomp->w = ccl::make_float4(shear.x, 0.0f, 0.0f, scale.z);
//}

// void svm_eval_nodes(ccl::KernelGlobalsCPU *kg, ccl::uint prim)
//{
//  // ccl::KERNEL_FEATURE_NODE_MASK_SURFACE & ~ccl::KERNEL_FEATURE_NODE_RAYTRACE
//  /*  KernelGlobals kg, ConstIntegratorGenericState state, ccl_private ShaderData *sd,
//        ccl_global float *render_buffer,
//        uint32_t path_flag*/
//
//  ccl::ShaderData sd;
//  uint32_t path_flag = 0;
//  ccl::svm_eval_nodes<ccl::KERNEL_FEATURE_NODE_MASK_SURFACE, ccl::SHADER_TYPE_SURFACE>(kg,
//  INTEGRATOR_STATE_NULL, sd, NULL, path_flag);
//}

ccl::uint4 read_node(ccl::KernelGlobalsCPU *kg, int *offset)
{
  ccl::uint4 node = kernel_data_fetch(svm_nodes, *offset);
  (*offset)++;
  return node;
}

void svm_unpack_node_uchar4(ccl::uint i, ccl::uint *x, ccl::uint *y, ccl::uint *z, ccl::uint *w)
{
  *x = (i & 0xFF);
  *y = ((i >> 8) & 0xFF);
  *z = ((i >> 16) & 0xFF);
  *w = ((i >> 24) & 0xFF);
}

bool stack_valid(ccl::uint a)
{
  return a != (ccl::uint)SVM_STACK_INVALID;
}

float stack_load_float(float *stack, uint a)
{
  return stack[a];
}

void stack_store_float(float *stack, uint a, float f)
{
  stack[a] = f;
}

float stack_load_float_default(float *stack, uint a, uint value)
{
  return (a == (uint)SVM_STACK_INVALID) ? ccl::__uint_as_float(value) : stack_load_float(stack, a);
}

ccl::float3 stack_load_float3(float *stack, uint a)
{
  float *stack_a = stack + a;
  return ccl::make_float3(stack_a[0], stack_a[1], stack_a[2]);
}

float linear_rgb_to_gray(ccl::KernelGlobalsCPU *kg, ccl::float3 c)
{
  return ccl::dot(c, ccl::float4_to_float3(kernel_data.film.rgb_to_y));
}

int svm_node_closure_bsdf(ccl::KernelGlobalsCPU *kg,
                          // ccl_private ShaderData *sd,
                          float *stack,
                          ccl::uint4 node,
                          uint32_t path_flag,
                          int offset,
                          int sd_flag,
                          ccl::float3 sd_svm_closure_weight,
                          mini::Material::SP mini_material)
{

  /////////////////////////////////
  uint type, param1_offset, param2_offset;

  uint mix_weight_offset;
  svm_unpack_node_uchar4(node.y, &type, &param1_offset, &param2_offset, &mix_weight_offset);
  float mix_weight = (stack_valid(mix_weight_offset) ? stack_load_float(stack, mix_weight_offset) :
                                                       1.0f);

  /* note we read this extra node before weight check, so offset is added */
  ccl::uint4 data_node = read_node(kg, &offset);

  /* Only compute BSDF for surfaces, transparent variable is shared with volume extinction. */
  // IF_KERNEL_NODES_FEATURE(BSDF)
  //{
  //  if ((shader_type != SHADER_TYPE_SURFACE) || mix_weight == 0.0f) {
  //    return svm_node_closure_bsdf_skip(kg, offset, type);
  //  }
  //}
  // else
  //{
  //  return svm_node_closure_bsdf_skip(kg, offset, type);
  //}

  // float3 N = stack_valid(data_node.x) ? stack_load_float3(stack, data_node.x) : sd->N;
  // if (!(sd->type & PRIMITIVE_CURVE)) {
  //  N = ensure_valid_reflection(sd->Ng, sd->I, N);
  //}

  float param1 = (stack_valid(param1_offset)) ? stack_load_float(stack, param1_offset) :
                                                ccl::__uint_as_float(node.z);
  float param2 = (stack_valid(param2_offset)) ? stack_load_float(stack, param2_offset) :
                                                ccl::__uint_as_float(node.w);

  switch (type) {
#ifdef __PRINCIPLED__
    case ccl::CLOSURE_BSDF_PRINCIPLED_ID: {
      uint specular_offset, roughness_offset, specular_tint_offset, anisotropic_offset,
          sheen_offset, sheen_tint_offset, clearcoat_offset, clearcoat_roughness_offset,
          eta_offset, transmission_offset, anisotropic_rotation_offset,
          transmission_roughness_offset;
      ccl::uint4 data_node2 = read_node(kg, &offset);

      ccl::float3 T = stack_load_float3(stack, data_node.y);
      svm_unpack_node_uchar4(data_node.z,
                             &specular_offset,
                             &roughness_offset,
                             &specular_tint_offset,
                             &anisotropic_offset);
      svm_unpack_node_uchar4(data_node.w,
                             &sheen_offset,
                             &sheen_tint_offset,
                             &clearcoat_offset,
                             &clearcoat_roughness_offset);
      svm_unpack_node_uchar4(data_node2.x,
                             &eta_offset,
                             &transmission_offset,
                             &anisotropic_rotation_offset,
                             &transmission_roughness_offset);

      // get Disney principled parameters
      float metallic = param1;
      float subsurface = param2;
      float specular = stack_load_float(stack, specular_offset);
      float roughness = stack_load_float(stack, roughness_offset);
      float specular_tint = stack_load_float(stack, specular_tint_offset);
      float anisotropic = stack_load_float(stack, anisotropic_offset);
      float sheen = stack_load_float(stack, sheen_offset);
      float sheen_tint = stack_load_float(stack, sheen_tint_offset);
      float clearcoat = stack_load_float(stack, clearcoat_offset);
      float clearcoat_roughness = stack_load_float(stack, clearcoat_roughness_offset);
      float transmission = stack_load_float(stack, transmission_offset);
      float anisotropic_rotation = stack_load_float(stack, anisotropic_rotation_offset);
      float transmission_roughness = stack_load_float(stack, transmission_roughness_offset);
      float eta = fmaxf(stack_load_float(stack, eta_offset), 1e-5f);

      ccl::ClosureType distribution = (ccl::ClosureType)data_node2.y;
      ccl::ClosureType subsurface_method = (ccl::ClosureType)data_node2.z;

      /* rotate tangent */
      // if (anisotropic_rotation != 0.0f)
      //  T = rotate_around_axis(T, N, anisotropic_rotation * M_2PI_F);

      /* calculate ior */
      float ior = (sd_flag & ccl::SD_BACKFACING) ? 1.0f / eta : eta;

      // calculate fresnel for refraction
      // float cosNO = dot(N, sd->I);
      // float fresnel = fresnel_dielectric_cos(cosNO, ior);

      // calculate weights of the diffuse and specular part
      float diffuse_weight = (1.0f - ccl::saturatef(metallic)) *
                             (1.0f - ccl::saturatef(transmission));

      float final_transmission = ccl::saturatef(transmission) * (1.0f - ccl::saturatef(metallic));
      float specular_weight = (1.0f - final_transmission);

      // get the base color
      ccl::uint4 data_base_color = read_node(kg, &offset);
      ccl::float3 base_color = stack_valid(data_base_color.x) ?
                                   stack_load_float3(stack, data_base_color.x) :
                                   ccl::make_float3(ccl::__uint_as_float(data_base_color.y),
                                                    ccl::__uint_as_float(data_base_color.z),
                                                    ccl::__uint_as_float(data_base_color.w));

      // get the additional clearcoat normal and subsurface scattering radius
      ccl::uint4 data_cn_ssr = read_node(kg, &offset);
      // ccl::float3 clearcoat_normal = stack_valid(data_cn_ssr.x) ?
      //                              stack_load_float3(stack, data_cn_ssr.x) :
      //                              sd->N;
      // if (!(sd->type & PRIMITIVE_CURVE)) {
      //  clearcoat_normal = ensure_valid_reflection(sd->Ng, sd->I, clearcoat_normal);
      //}
      ccl::float3 subsurface_radius = stack_valid(data_cn_ssr.y) ?
                                          stack_load_float3(stack, data_cn_ssr.y) :
                                          ccl::make_float3(1.0f, 1.0f, 1.0f);
      float subsurface_ior = stack_valid(data_cn_ssr.z) ? stack_load_float(stack, data_cn_ssr.z) :
                                                          1.4f;
      float subsurface_anisotropy = stack_valid(data_cn_ssr.w) ?
                                        stack_load_float(stack, data_cn_ssr.w) :
                                        0.0f;

      // get the subsurface color
      ccl::uint4 data_subsurface_color = read_node(kg, &offset);
      ccl::float3 subsurface_color = stack_valid(data_subsurface_color.x) ?
                                         stack_load_float3(stack, data_subsurface_color.x) :
                                         ccl::make_float3(
                                             ccl::__uint_as_float(data_subsurface_color.y),
                                             ccl::__uint_as_float(data_subsurface_color.z),
                                             ccl::__uint_as_float(data_subsurface_color.w));

      ccl::float3 weight = sd_svm_closure_weight * mix_weight;

#  ifdef __SUBSURFACE__
      ccl::float3 mixed_ss_base_color = subsurface_color * subsurface +
                                        base_color * (1.0f - subsurface);
      ccl::float3 subsurf_weight = weight * mixed_ss_base_color * diffuse_weight;

      /* disable in case of diffuse ancestor, can't see it well then and
       * adds considerably noise due to probabilities of continuing path
       * getting lower and lower */
      if (path_flag & ccl::PATH_RAY_DIFFUSE_ANCESTOR) {
        subsurface = 0.0f;

        /* need to set the base color in this case such that the
         * rays get the correctly mixed color after transmitting
         * the object */
        base_color = mixed_ss_base_color;
      }

      /* diffuse */
      if (fabsf(average(mixed_ss_base_color)) > CLOSURE_WEIGHT_CUTOFF) {
        if (subsurface <= CLOSURE_WEIGHT_CUTOFF && diffuse_weight > CLOSURE_WEIGHT_CUTOFF) {
          ccl::float3 diff_weight = weight * base_color * diffuse_weight;

          // ccl_private PrincipledDiffuseBsdf *bsdf = (ccl_private PrincipledDiffuseBsdf *)
          //    bsdf_alloc(sd, sizeof(PrincipledDiffuseBsdf), diff_weight);

          // if (bsdf) {
          //  bsdf->N = N;
          //  bsdf->roughness = roughness;

          //  /* setup bsdf */
          //  sd->flag |= bsdf_principled_diffuse_setup(bsdf, PRINCIPLED_DIFFUSE_FULL);
          //}
        }
        else if (subsurface > CLOSURE_WEIGHT_CUTOFF) {
          // ccl_private Bssrdf *bssrdf = bssrdf_alloc(sd, subsurf_weight);

          // if (bssrdf) {
          //  bssrdf->radius = subsurface_radius * subsurface;
          //  bssrdf->albedo = mixed_ss_base_color;
          //  bssrdf->N = N;
          //  bssrdf->roughness = roughness;

          //  /* Clamps protecting against bad/extreme and non physical values. */
          //  subsurface_ior = clamp(subsurface_ior, 1.01f, 3.8f);
          //  bssrdf->anisotropy = clamp(subsurface_anisotropy, 0.0f, 0.9f);

          //  /* setup bsdf */
          //  sd->flag |= bssrdf_setup(sd, bssrdf, subsurface_method, subsurface_ior);
          //}
        }
      }
#  else
      /* diffuse */
      if (diffuse_weight > CLOSURE_WEIGHT_CUTOFF) {
        float3 diff_weight = weight * base_color * diffuse_weight;

        // ccl_private PrincipledDiffuseBsdf *bsdf = (ccl_private PrincipledDiffuseBsdf
        // *)bsdf_alloc(
        //    sd, sizeof(PrincipledDiffuseBsdf), diff_weight);

        // if (bsdf) {
        //  bsdf->N = N;
        //  bsdf->roughness = roughness;

        //  /* setup bsdf */
        //  sd->flag |= bsdf_principled_diffuse_setup(bsdf, PRINCIPLED_DIFFUSE_FULL);
        //}
      }
#  endif

      /* sheen */
      if (diffuse_weight > CLOSURE_WEIGHT_CUTOFF && sheen > CLOSURE_WEIGHT_CUTOFF) {
        float m_cdlum = linear_rgb_to_gray(kg, base_color);
        ccl::float3 m_ctint = m_cdlum > 0.0f ?
                                  base_color / m_cdlum :
                                  ccl::make_float3(
                                      1.0f, 1.0f, 1.0f);  // normalize lum. to isolate hue+sat

        /* color of the sheen component */
        ccl::float3 sheen_color = ccl::make_float3(1.0f, 1.0f, 1.0f) * (1.0f - sheen_tint) +
                                  m_ctint * sheen_tint;

        ccl::float3 sheen_weight = weight * sheen * sheen_color * diffuse_weight;

        // ccl_private PrincipledSheenBsdf *bsdf = (ccl_private PrincipledSheenBsdf *)bsdf_alloc(
        //    sd, sizeof(PrincipledSheenBsdf), sheen_weight);

        // if (bsdf) {
        //  bsdf->N = N;

        //  /* setup bsdf */
        //  sd->flag |= bsdf_principled_sheen_setup(sd, bsdf);
        //}
      }

      /* specular reflection */
#  ifdef __CAUSTICS_TRICKS__
      if (kernel_data.integrator.caustics_reflective || (path_flag & ccl::PATH_RAY_DIFFUSE) == 0) {
#  endif
        if (specular_weight > CLOSURE_WEIGHT_CUTOFF &&
            (specular > CLOSURE_WEIGHT_CUTOFF || metallic > CLOSURE_WEIGHT_CUTOFF)) {
          ccl::float3 spec_weight = weight * specular_weight;

          // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
          //    sd, sizeof(MicrofacetBsdf), spec_weight);
          // ccl_private MicrofacetExtra *extra =
          //    (bsdf != NULL) ?
          //        (ccl_private MicrofacetExtra *)closure_alloc_extra(sd, sizeof(MicrofacetExtra))
          //        : NULL;

          // if (bsdf && extra) {
          //  bsdf->N = N;
          //  bsdf->ior = (2.0f / (1.0f - safe_sqrtf(0.08f * specular))) - 1.0f;
          //  bsdf->T = T;
          //  bsdf->extra = extra;

          //  float aspect = safe_sqrtf(1.0f - anisotropic * 0.9f);
          //  float r2 = roughness * roughness;

          //  bsdf->alpha_x = r2 / aspect;
          //  bsdf->alpha_y = r2 * aspect;

          //  float m_cdlum = 0.3f * base_color.x + 0.6f * base_color.y +
          //                  0.1f * base_color.z;  // luminance approx.
          //  float3 m_ctint = m_cdlum > 0.0f ?
          //                       base_color / m_cdlum :
          //                       make_float3(
          //                           1.0f, 1.0f, 1.0f);  // normalize lum. to isolate hue+sat
          //  float3 tmp_col = make_float3(1.0f, 1.0f, 1.0f) * (1.0f - specular_tint) +
          //                   m_ctint * specular_tint;

          //  bsdf->extra->cspec0 = (specular * 0.08f * tmp_col) * (1.0f - metallic) +
          //                        base_color * metallic;
          //  bsdf->extra->color = base_color;
          //  bsdf->extra->clearcoat = 0.0f;

          //  /* setup bsdf */
          //  if (distribution == CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID ||
          //      roughness <= 0.075f) /* use single-scatter GGX */
          //    sd->flag |= bsdf_microfacet_ggx_fresnel_setup(bsdf, sd);
          //  else /* use multi-scatter GGX */
          //    sd->flag |= bsdf_microfacet_multi_ggx_fresnel_setup(bsdf, sd);
          //}
        }
#  ifdef __CAUSTICS_TRICKS__
      }
#  endif

      /* BSDF */
#  ifdef __CAUSTICS_TRICKS__
      if (kernel_data.integrator.caustics_reflective ||
          kernel_data.integrator.caustics_refractive || (path_flag & ccl::PATH_RAY_DIFFUSE) == 0) {
#  endif
        if (final_transmission > CLOSURE_WEIGHT_CUTOFF) {
          ccl::float3 glass_weight = weight * final_transmission;
          ccl::float3 cspec0 = base_color * specular_tint +
                               ccl::make_float3(1.0f, 1.0f, 1.0f) * (1.0f - specular_tint);

          if (roughness <= 5e-2f ||
              distribution ==
                  ccl::CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) { /* use single-scatter GGX */
            float refl_roughness = roughness;

            /* reflection */
#  ifdef __CAUSTICS_TRICKS__
            if (kernel_data.integrator.caustics_reflective ||
                (path_flag & ccl::PATH_RAY_DIFFUSE) == 0)
#  endif
            {
              // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
              //    sd, sizeof(MicrofacetBsdf), glass_weight * fresnel);
              // ccl_private MicrofacetExtra *extra =
              //    (bsdf != NULL) ? (ccl_private MicrofacetExtra *)closure_alloc_extra(
              //                         sd, sizeof(MicrofacetExtra)) :
              //                     NULL;

              // if (bsdf && extra) {
              //  bsdf->N = N;
              //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
              //  bsdf->extra = extra;

              //  bsdf->alpha_x = refl_roughness * refl_roughness;
              //  bsdf->alpha_y = refl_roughness * refl_roughness;
              //  bsdf->ior = ior;

              //  bsdf->extra->color = base_color;
              //  bsdf->extra->cspec0 = cspec0;
              //  bsdf->extra->clearcoat = 0.0f;

              //  /* setup bsdf */
              //  sd->flag |= bsdf_microfacet_ggx_fresnel_setup(bsdf, sd);
              //}
            }

            /* refraction */
#  ifdef __CAUSTICS_TRICKS__
            if (kernel_data.integrator.caustics_refractive ||
                (path_flag & ccl::PATH_RAY_DIFFUSE) == 0)
#  endif
            {
              /* This is to prevent MNEE from receiving a null BSDF. */
              // float refraction_fresnel = fmaxf(0.0001f, 1.0f - fresnel);
              // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
              //    sd, sizeof(MicrofacetBsdf), base_color * glass_weight * refraction_fresnel);
              // if (bsdf) {
              //  bsdf->N = N;
              //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
              //  bsdf->extra = NULL;

              //  if (distribution == CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)
              //    transmission_roughness = 1.0f - (1.0f - refl_roughness) *
              //                                        (1.0f - transmission_roughness);
              //  else
              //    transmission_roughness = refl_roughness;

              //  bsdf->alpha_x = transmission_roughness * transmission_roughness;
              //  bsdf->alpha_y = transmission_roughness * transmission_roughness;
              //  bsdf->ior = ior;

              //  /* setup bsdf */
              //  sd->flag |= bsdf_microfacet_ggx_refraction_setup(bsdf);
              //}
            }
          }
          else { /* use multi-scatter GGX */
            // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
            //    sd, sizeof(MicrofacetBsdf), glass_weight);
            // ccl_private MicrofacetExtra *extra =
            //    (bsdf != NULL) ? (ccl_private MicrofacetExtra *)closure_alloc_extra(
            //                         sd, sizeof(MicrofacetExtra)) :
            //                     NULL;

            // if (bsdf && extra) {
            //  bsdf->N = N;
            //  bsdf->extra = extra;
            //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);

            //  bsdf->alpha_x = roughness * roughness;
            //  bsdf->alpha_y = roughness * roughness;
            //  bsdf->ior = ior;

            //  bsdf->extra->color = base_color;
            //  bsdf->extra->cspec0 = cspec0;
            //  bsdf->extra->clearcoat = 0.0f;

            //  /* setup bsdf */
            //  sd->flag |= bsdf_microfacet_multi_ggx_glass_fresnel_setup(bsdf, sd);
            //}
          }
        }
#  ifdef __CAUSTICS_TRICKS__
      }
#  endif

      /* clearcoat */
#  ifdef __CAUSTICS_TRICKS__
      if (kernel_data.integrator.caustics_reflective || (path_flag & ccl::PATH_RAY_DIFFUSE) == 0) {
#  endif
        if (clearcoat > CLOSURE_WEIGHT_CUTOFF) {
          // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
          //    sd, sizeof(MicrofacetBsdf), weight);
          // ccl_private MicrofacetExtra *extra =
          //    (bsdf != NULL) ?
          //        (ccl_private MicrofacetExtra *)closure_alloc_extra(sd, sizeof(MicrofacetExtra))
          //        : NULL;

          // if (bsdf && extra) {
          //  bsdf->N = clearcoat_normal;
          //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
          //  bsdf->ior = 1.5f;
          //  bsdf->extra = extra;

          //  bsdf->alpha_x = clearcoat_roughness * clearcoat_roughness;
          //  bsdf->alpha_y = clearcoat_roughness * clearcoat_roughness;

          //  bsdf->extra->color = make_float3(0.0f, 0.0f, 0.0f);
          //  bsdf->extra->cspec0 = make_float3(0.04f, 0.04f, 0.04f);
          //  bsdf->extra->clearcoat = clearcoat;

          //  /* setup bsdf */
          //  sd->flag |= bsdf_microfacet_ggx_clearcoat_setup(bsdf, sd);
          //}
        }
#  ifdef __CAUSTICS_TRICKS__
      }
#  endif
      /////////////////////////////////////////////////////////
      // vec3f baseColor{.5f, .5f, .5f};
      // float metallic{0.f};
      // float roughness{0.f};
      // float transmission{0.f};
      // float ior{1.45f};
      mini_material->baseColor[0] = base_color[0];
      mini_material->baseColor[1] = base_color[1];
      mini_material->baseColor[2] = base_color[2];

      mini_material->metallic = metallic;
      mini_material->roughness = roughness;
      mini_material->transmission = transmission;
      mini_material->ior = ior;
      /////////////////////////////////////////////////////////

      break;
    }
#endif /* __PRINCIPLED__ */
    case ccl::CLOSURE_BSDF_DIFFUSE_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private OrenNayarBsdf *bsdf = (ccl_private OrenNayarBsdf *)bsdf_alloc(
      //    sd, sizeof(OrenNayarBsdf), weight);

      // if (bsdf) {
      //  bsdf->N = N;

      float roughness = param1;

      //  if (roughness == 0.0f) {
      //    sd->flag |= bsdf_diffuse_setup((ccl_private DiffuseBsdf *)bsdf);
      //  }
      //  else {
      //    bsdf->roughness = roughness;
      //    sd->flag |= bsdf_oren_nayar_setup(bsdf);
      //  }
      //}

      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->roughness = roughness;
      /////////////////////////////////////////////////////////

      break;
    }
    case ccl::CLOSURE_BSDF_TRANSLUCENT_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private DiffuseBsdf *bsdf = (ccl_private DiffuseBsdf *)bsdf_alloc(
      //    sd, sizeof(DiffuseBsdf), weight);

      // if (bsdf) {
      //  bsdf->N = N;
      //  sd->flag |= bsdf_translucent_setup(bsdf);
      //}
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->transmission = 1.0f;
      /////////////////////////////////////////////////////////
      break;
    }
    case ccl::CLOSURE_BSDF_TRANSPARENT_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // bsdf_transparent_setup(sd, weight, path_flag);
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->transmission = 1.0f;
      /////////////////////////////////////////////////////////
      break;
    }
    case ccl::CLOSURE_BSDF_REFLECTION_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_GGX_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case ccl::CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID: {
#ifdef __CAUSTICS_TRICKS__
      if (!kernel_data.integrator.caustics_reflective && (path_flag & ccl::PATH_RAY_DIFFUSE))
        break;
#endif
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      //    sd, sizeof(MicrofacetBsdf), weight);

      // if (!bsdf) {
      //  break;
      //}

      float roughness = ccl::sqr(param1);

      // bsdf->N = N;
      // bsdf->ior = 0.0f;
      float ior = 0.0f;
      // bsdf->extra = NULL;

      // if (data_node.y == SVM_STACK_INVALID) {
      //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
      //  bsdf->alpha_x = roughness;
      //  bsdf->alpha_y = roughness;
      //}
      // else {
      //  bsdf->T = stack_load_float3(stack, data_node.y);

      //  /* rotate tangent */
      //  float rotation = stack_load_float(stack, data_node.z);
      //  if (rotation != 0.0f)
      //    bsdf->T = rotate_around_axis(bsdf->T, bsdf->N, rotation * M_2PI_F);

      //  /* compute roughness */
      //  float anisotropy = clamp(param2, -0.99f, 0.99f);
      //  if (anisotropy < 0.0f) {
      //    bsdf->alpha_x = roughness / (1.0f + anisotropy);
      //    bsdf->alpha_y = roughness * (1.0f + anisotropy);
      //  }
      //  else {
      //    bsdf->alpha_x = roughness * (1.0f - anisotropy);
      //    bsdf->alpha_y = roughness / (1.0f - anisotropy);
      //  }
      //}

      ///* setup bsdf */
      // if (type == CLOSURE_BSDF_REFLECTION_ID)
      //  sd->flag |= bsdf_reflection_setup(bsdf);
      // else if (type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID)
      //  sd->flag |= bsdf_microfacet_beckmann_setup(bsdf);
      // else if (type == CLOSURE_BSDF_MICROFACET_GGX_ID)
      //  sd->flag |= bsdf_microfacet_ggx_setup(bsdf);
      // else if (type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) {
      //  kernel_assert(stack_valid(data_node.w));
      //  bsdf->extra = (ccl_private MicrofacetExtra *)closure_alloc_extra(sd,
      //                                                                   sizeof(MicrofacetExtra));
      //  if (bsdf->extra) {
      //    bsdf->extra->color = stack_load_float3(stack, data_node.w);
      //    bsdf->extra->cspec0 = make_float3(0.0f, 0.0f, 0.0f);
      //    bsdf->extra->clearcoat = 0.0f;
      //    sd->flag |= bsdf_microfacet_multi_ggx_setup(bsdf);
      //  }
      //}
      // else {
      //  sd->flag |= bsdf_ashikhmin_shirley_setup(bsdf);
      //}

      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->roughness = roughness;
      mini_material->ior = ior;
      /////////////////////////////////////////////////////////

      break;
    }
    case ccl::CLOSURE_BSDF_REFRACTION_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID: {
#ifdef __CAUSTICS_TRICKS__
      if (!kernel_data.integrator.caustics_refractive && (path_flag & ccl::PATH_RAY_DIFFUSE))
        break;
#endif
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      //    sd, sizeof(MicrofacetBsdf), weight);

      // if (bsdf) {
      //  bsdf->N = N;
      //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
      //  bsdf->extra = NULL;

      float eta = fmaxf(param2, 1e-5f);
      eta = (sd_flag & ccl::SD_BACKFACING) ? 1.0f / eta : eta;

      float roughness = 0.0f;

      //  /* setup bsdf */
      if (type == ccl::CLOSURE_BSDF_REFRACTION_ID) {
        //    bsdf->alpha_x = 0.0f;
        //    bsdf->alpha_y = 0.0f;
        //    bsdf->ior = eta;

        //    sd->flag |= bsdf_refraction_setup(bsdf);
      }
      else {
        roughness = ccl::sqr(param1);
        //    bsdf->alpha_x = roughness;
        //    bsdf->alpha_y = roughness;
        //    bsdf->ior = eta;

        //    if (type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID)
        //      sd->flag |= bsdf_microfacet_beckmann_refraction_setup(bsdf);
        //    else
        //      sd->flag |= bsdf_microfacet_ggx_refraction_setup(bsdf);
        //  }
      }

      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->transmission = roughness;
      mini_material->ior = eta;
      /////////////////////////////////////////////////////////

      break;
    }
    case ccl::CLOSURE_BSDF_SHARP_GLASS_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case ccl::CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID: {
#ifdef __CAUSTICS_TRICKS__
      if (!kernel_data.integrator.caustics_reflective &&
          !kernel_data.integrator.caustics_refractive && (path_flag & ccl::PATH_RAY_DIFFUSE)) {
        break;
      }
#endif
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;

      /* index of refraction */
      float eta = fmaxf(param2, 1e-5f);
      eta = (sd_flag & ccl::SD_BACKFACING) ? 1.0f / eta : eta;

      /* fresnel */
      // float cosNO = dot(N, sd->I);
      // float fresnel = fresnel_dielectric_cos(cosNO, eta);
      float roughness = ccl::sqr(param1);

      /* reflection */
#ifdef __CAUSTICS_TRICKS__
      if (kernel_data.integrator.caustics_reflective || (path_flag & ccl::PATH_RAY_DIFFUSE) == 0)
#endif
      {
        // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
        //    sd, sizeof(MicrofacetBsdf), weight * fresnel);

        // if (bsdf) {
        //  bsdf->N = N;
        //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
        //  bsdf->extra = NULL;
        //  svm_node_glass_setup(sd, bsdf, type, eta, roughness, false);
        //}
      }

      /* refraction */
#ifdef __CAUSTICS_TRICKS__
      if (kernel_data.integrator.caustics_refractive || (path_flag & ccl::PATH_RAY_DIFFUSE) == 0)
#endif
      {
        /* This is to prevent MNEE from receiving a null BSDF. */
        // float refraction_fresnel = fmaxf(0.0001f, 1.0f - fresnel);
        // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
        //    sd, sizeof(MicrofacetBsdf), weight * refraction_fresnel);

        // if (bsdf) {
        //  bsdf->N = N;
        //  bsdf->T = make_float3(0.0f, 0.0f, 0.0f);
        //  bsdf->extra = NULL;
        //  svm_node_glass_setup(sd, bsdf, type, eta, roughness, true);
        //}
      }

      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->transmission = roughness;
      mini_material->ior = eta;
      /////////////////////////////////////////////////////////

      break;
    }
    case ccl::CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID: {
#ifdef __CAUSTICS_TRICKS__
      if (!kernel_data.integrator.caustics_reflective &&
          !kernel_data.integrator.caustics_refractive && (path_flag & ccl::PATH_RAY_DIFFUSE))
        break;
#endif
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      //    sd, sizeof(MicrofacetBsdf), weight);
      // if (!bsdf) {
      //  break;
      //}

      // ccl_private MicrofacetExtra *extra = (ccl_private MicrofacetExtra *)closure_alloc_extra(
      //    sd, sizeof(MicrofacetExtra));
      // if (!extra) {
      //  break;
      //}

      // bsdf->N = N;
      // bsdf->extra = extra;
      // bsdf->T = make_float3(0.0f, 0.0f, 0.0f);

      float roughness = ccl::sqr(param1);
      // bsdf->alpha_x = roughness;
      // bsdf->alpha_y = roughness;
      float eta = fmaxf(param2, 1e-5f);
      float ior = (sd_flag & ccl::SD_BACKFACING) ? 1.0f / eta : eta;

      // kernel_assert(stack_valid(data_node.z));
      // bsdf->extra->color = stack_load_float3(stack, data_node.z);
      // bsdf->extra->cspec0 = make_float3(0.0f, 0.0f, 0.0f);
      // bsdf->extra->clearcoat = 0.0f;

      ///* setup bsdf */
      // sd->flag |= bsdf_microfacet_multi_ggx_glass_setup(bsdf);
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->transmission = roughness;
      mini_material->ior = ior;
      /////////////////////////////////////////////////////////

      break;
    }
    case ccl::CLOSURE_BSDF_ASHIKHMIN_VELVET_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private VelvetBsdf *bsdf = (ccl_private VelvetBsdf *)bsdf_alloc(
      //    sd, sizeof(VelvetBsdf), weight);

      // if (bsdf) {
      //  bsdf->N = N;

      //  bsdf->sigma = saturatef(param1);
      //  sd->flag |= bsdf_ashikhmin_velvet_setup(bsdf);
      //}
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];
      /////////////////////////////////////////////////////////
      break;
    }
    case ccl::CLOSURE_BSDF_GLOSSY_TOON_ID:
#ifdef __CAUSTICS_TRICKS__
      if (!kernel_data.integrator.caustics_reflective && (path_flag & ccl::PATH_RAY_DIFFUSE))
        break;
      ATTR_FALLTHROUGH;
#endif
    case ccl::CLOSURE_BSDF_DIFFUSE_TOON_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private ToonBsdf *bsdf = (ccl_private ToonBsdf *)bsdf_alloc(
      //    sd, sizeof(ToonBsdf), weight);

      // if (bsdf) {
      //  bsdf->N = N;
      //  bsdf->size = param1;
      //  bsdf->smooth = param2;

      //  if (type == CLOSURE_BSDF_DIFFUSE_TOON_ID)
      //    sd->flag |= bsdf_diffuse_toon_setup(bsdf);
      //  else
      //    sd->flag |= bsdf_glossy_toon_setup(bsdf);
      //}
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];
      /////////////////////////////////////////////////////////
      break;
    }
#ifdef __HAIR__
    case ccl::CLOSURE_BSDF_HAIR_PRINCIPLED_ID: {
      ccl::uint4 data_node2 = read_node(kg, &offset);
      ccl::uint4 data_node3 = read_node(kg, &offset);
      ccl::uint4 data_node4 = read_node(kg, &offset);

      ccl::float3 weight = sd_svm_closure_weight * mix_weight;

      uint offset_ofs, ior_ofs, color_ofs, parametrization;
      svm_unpack_node_uchar4(data_node.y, &offset_ofs, &ior_ofs, &color_ofs, &parametrization);
      float alpha = stack_load_float_default(stack, offset_ofs, data_node.z);
      float ior = stack_load_float_default(stack, ior_ofs, data_node.w);

      uint coat_ofs, melanin_ofs, melanin_redness_ofs, absorption_coefficient_ofs;
      svm_unpack_node_uchar4(data_node2.x,
                             &coat_ofs,
                             &melanin_ofs,
                             &melanin_redness_ofs,
                             &absorption_coefficient_ofs);

      uint tint_ofs, random_ofs, random_color_ofs, random_roughness_ofs;
      svm_unpack_node_uchar4(
          data_node3.x, &tint_ofs, &random_ofs, &random_color_ofs, &random_roughness_ofs);

      // const AttributeDescriptor attr_descr_random = find_attribute(kg, sd, data_node4.y);
      float random = 0.0f;
      // if (attr_descr_random.offset != ATTR_STD_NOT_FOUND) {
      //  random = primitive_surface_attribute_float(kg, sd, attr_descr_random, NULL, NULL);
      //}
      // else {
      //  random = stack_load_float_default(stack, random_ofs, data_node3.y);
      //}

      // ccl_private PrincipledHairBSDF *bsdf = (ccl_private PrincipledHairBSDF *)bsdf_alloc(
      //    sd, sizeof(PrincipledHairBSDF), weight);
      // if (bsdf) {
      //  ccl_private PrincipledHairExtra *extra = (ccl_private PrincipledHairExtra *)
      //      closure_alloc_extra(sd, sizeof(PrincipledHairExtra));

      //  if (!extra)
      //    break;

      //  /* Random factors range: [-randomization/2, +randomization/2]. */
      float random_roughness = stack_load_float_default(stack, random_roughness_ofs, data_node3.w);
      float factor_random_roughness = 1.0f + 2.0f * (random - 0.5f) * random_roughness;
      float roughness = param1 * factor_random_roughness;
      float radial_roughness = param2 * factor_random_roughness;

      //  /* Remap Coat value to [0, 100]% of Roughness. */
      //  float coat = stack_load_float_default(stack, coat_ofs, data_node2.y);
      //  float m0_roughness = 1.0f - clamp(coat, 0.0f, 1.0f);

      //  bsdf->N = N;
      //  bsdf->v = roughness;
      //  bsdf->s = radial_roughness;
      //  bsdf->m0_roughness = m0_roughness;
      //  bsdf->alpha = alpha;
      //  bsdf->eta = ior;
      //  bsdf->extra = extra;

      //  switch (parametrization) {
      //    case NODE_PRINCIPLED_HAIR_DIRECT_ABSORPTION: {
      //      float3 absorption_coefficient = stack_load_float3(stack, absorption_coefficient_ofs);
      //      bsdf->sigma = absorption_coefficient;
      //      break;
      //    }
      //    case NODE_PRINCIPLED_HAIR_PIGMENT_CONCENTRATION: {
      //      float melanin = stack_load_float_default(stack, melanin_ofs, data_node2.z);
      //      float melanin_redness = stack_load_float_default(
      //          stack, melanin_redness_ofs, data_node2.w);

      //      /* Randomize melanin. */
      //      float random_color = stack_load_float_default(stack, random_color_ofs, data_node3.z);
      //      random_color = clamp(random_color, 0.0f, 1.0f);
      //      float factor_random_color = 1.0f + 2.0f * (random - 0.5f) * random_color;
      //      melanin *= factor_random_color;

      //      /* Map melanin 0..inf from more perceptually linear 0..1. */
      //      melanin = -logf(fmaxf(1.0f - melanin, 0.0001f));

      //      /* Benedikt Bitterli's melanin ratio remapping. */
      //      float eumelanin = melanin * (1.0f - melanin_redness);
      //      float pheomelanin = melanin * melanin_redness;
      //      float3 melanin_sigma = bsdf_principled_hair_sigma_from_concentration(eumelanin,
      //                                                                           pheomelanin);

      //      /* Optional tint. */
      //      float3 tint = stack_load_float3(stack, tint_ofs);
      //      float3 tint_sigma = bsdf_principled_hair_sigma_from_reflectance(tint,
      //                                                                      radial_roughness);

      //      bsdf->sigma = melanin_sigma + tint_sigma;
      //      break;
      //    }
      //    case NODE_PRINCIPLED_HAIR_REFLECTANCE: {
      //      float3 color = stack_load_float3(stack, color_ofs);
      //      bsdf->sigma = bsdf_principled_hair_sigma_from_reflectance(color, radial_roughness);
      //      break;
      //    }
      //    default: {
      //      /* Fallback to brownish hair, same as defaults for melanin. */
      //      kernel_assert(!"Invalid Principled Hair parametrization!");
      //      bsdf->sigma = bsdf_principled_hair_sigma_from_concentration(0.0f, 0.8054375f);
      //      break;
      //    }
      //  }

      //  sd->flag |= bsdf_principled_hair_setup(sd, bsdf);
      //}
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->ior = ior;
      mini_material->transmission = alpha;
      mini_material->roughness = roughness;
      /////////////////////////////////////////////////////////
      break;
    }
    case ccl::CLOSURE_BSDF_HAIR_REFLECTION_ID:
    case ccl::CLOSURE_BSDF_HAIR_TRANSMISSION_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;

      // ccl_private HairBsdf *bsdf = (ccl_private HairBsdf *)bsdf_alloc(
      //    sd, sizeof(HairBsdf), weight);

      // if (bsdf) {
      //  bsdf->N = N;
      //  bsdf->roughness1 = param1;
      //  bsdf->roughness2 = param2;
      //  bsdf->offset = -stack_load_float(stack, data_node.z);

      //  if (stack_valid(data_node.y)) {
      //    bsdf->T = normalize(stack_load_float3(stack, data_node.y));
      //  }
      //  else if (!(sd->type & PRIMITIVE_CURVE)) {
      //    bsdf->T = normalize(sd->dPdv);
      //    bsdf->offset = 0.0f;
      //  }
      //  else
      //    bsdf->T = normalize(sd->dPdu);

      //  if (type == CLOSURE_BSDF_HAIR_REFLECTION_ID) {
      //    sd->flag |= bsdf_hair_reflection_setup(bsdf);
      //  }
      //  else {
      //    sd->flag |= bsdf_hair_transmission_setup(bsdf);
      //  }
      //}
      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->roughness = param1;
      /////////////////////////////////////////////////////////
      break;
    }
#endif /* __HAIR__ */

#ifdef __SUBSURFACE__
    case ccl::CLOSURE_BSSRDF_BURLEY_ID:
    case ccl::CLOSURE_BSSRDF_RANDOM_WALK_ID:
    case ccl::CLOSURE_BSSRDF_RANDOM_WALK_FIXED_RADIUS_ID: {
      ccl::float3 weight = sd_svm_closure_weight * mix_weight;
      // ccl_private Bssrdf *bssrdf = bssrdf_alloc(sd, weight);

      // if (bssrdf) {
      //  /* disable in case of diffuse ancestor, can't see it well then and
      //   * adds considerably noise due to probabilities of continuing path
      //   * getting lower and lower */
      //  if (path_flag & PATH_RAY_DIFFUSE_ANCESTOR)
      //    param1 = 0.0f;

      //  bssrdf->radius = stack_load_float3(stack, data_node.z) * param1;
      //  bssrdf->albedo = sd->svm_closure_weight;
      //  bssrdf->N = N;
      //  bssrdf->roughness = FLT_MAX;

      //  const float subsurface_ior = clamp(param2, 1.01f, 3.8f);
      //  const float subsurface_anisotropy = stack_load_float(stack, data_node.w);
      //  bssrdf->anisotropy = clamp(subsurface_anisotropy, 0.0f, 0.9f);

      //  sd->flag |= bssrdf_setup(sd, bssrdf, (ClosureType)type, subsurface_ior);
      //}

      /////////////////////////////////////////////////////////
      mini_material->baseColor[0] = weight[0];
      mini_material->baseColor[1] = weight[1];
      mini_material->baseColor[2] = weight[2];

      mini_material->roughness = FLT_MAX;
      /////////////////////////////////////////////////////////

      break;
    }
#endif
    default:
      break;
  }

  return offset;
}

float half_to_float(unsigned short h)
{
  float f;

  *((int *)&f) = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);

  return f;
}

void set_tex_interpolation(const ccl::TextureInfo &info, mini::Texture::SP mini_tex)
{
  if (info.interpolation == ccl::INTERPOLATION_NONE ||
      info.interpolation == ccl::INTERPOLATION_CLOSEST)
    mini_tex->filterMode = mini::Texture::FILTER_NEAREST;
  else
    mini_tex->filterMode = mini::Texture::FILTER_BILINEAR;
}

int svm_node_tex_image_byte4(ccl::KernelGlobalsCPU *kg,
                             float *stack,
                             // ccl::uint4 node,
                             int num_nodes,
                             int offset,
                             mini::Material::SP mini_material)
{
  // offset = svm_node_tex_image(kg, sd, stack, node, offset);
  // int num_nodes = (int)node.y;
  if (num_nodes <= 0) {
    int id = -num_nodes;
    const ccl::TextureInfo &info = kernel_data_fetch(texture_info, id);
    if (info.data) {
      mini_material->colorTexture = std::make_shared<mini::Texture>();
      set_tex_interpolation(info, mini_material->colorTexture);

      switch (info.data_type) {
        case ccl::IMAGE_DATA_TYPE_HALF: {
          // const float f = TextureInterpolator<ccl::half, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          ccl::half *data_in = (ccl::half *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              // size_t id = x + (size_t)y * info.width;
              // data_out[id] = half_to_float(data_in[id]);
              size_t id = x + (size_t)y * info.width;
              uchar value = (uchar)(half_to_float(data_in[id]) * 255.0f);
              data_out[id * 4 + 0] = value;
              data_out[id * 4 + 1] = value;
              data_out[id * 4 + 2] = value;
              data_out[id * 4 + 3] = 255;
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_BYTE: {
          // const float f = TextureInterpolator<uchar, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          uchar *data_in = (uchar *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              // data_out[id] = (float)data_in[id] / (float)255.0f;
              uchar value = data_in[id];
              data_out[id * 4 + 0] = value;
              data_out[id * 4 + 1] = value;
              data_out[id * 4 + 2] = value;
              data_out[id * 4 + 3] = 255;
            }
          }

          break;
        }
        case ccl::IMAGE_DATA_TYPE_USHORT: {
          // const float f = TextureInterpolator<uint16_t, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          uint16_t *data_in = (uint16_t *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              // data_out[id] = (float)data_in[id] / (float)65535.0f;
              uchar value = (uchar)((float)data_in[id] * 255.0f / 65535.0f);
              data_out[id * 4 + 0] = value;
              data_out[id * 4 + 1] = value;
              data_out[id * 4 + 2] = value;
              data_out[id * 4 + 3] = 255;
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_FLOAT: {
          // const float f = TextureInterpolator<float, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          float *data_in = (float *)info.data;

          // memcpy(mini_material->colorTexture->data.data(),
          //       (char *)info.data,
          //       mini_material->colorTexture->data.size());
#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              // data_out[id] = (float)data_in[id] / (float)65535.0f;
              uchar value = (uchar)(data_in[id] * 255.0f);
              data_out[id * 4 + 0] = value;
              data_out[id * 4 + 1] = value;
              data_out[id * 4 + 2] = value;
              data_out[id * 4 + 3] = 255;
            }
          }

          break;
        }
        case ccl::IMAGE_DATA_TYPE_HALF4: {
          // return TextureInterpolator<half4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          ccl::half *data_in = (ccl::half *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id * 4 + 0] = (uchar)(half_to_float(data_in[id * 4 + 0]) * 255.0f);
              data_out[id * 4 + 1] = (uchar)(half_to_float(data_in[id * 4 + 1]) * 255.0f);
              data_out[id * 4 + 2] = (uchar)(half_to_float(data_in[id * 4 + 2]) * 255.0f);
              data_out[id * 4 + 3] = (uchar)(half_to_float(data_in[id * 4 + 3]) * 255.0f);
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_BYTE4: {
          // return TextureInterpolator<uchar4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(char) * 4 * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          memcpy(mini_material->colorTexture->data.data(),
                 (char *)info.data,
                 mini_material->colorTexture->data.size());
          break;
        }
        case ccl::IMAGE_DATA_TYPE_USHORT4: {
          // return TextureInterpolator<ushort4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          uint16_t *data_in = (uint16_t *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id * 4 + 0] = (uchar)((float)data_in[id * 4 + 0] * 255.0f / 65535.0f);
              data_out[id * 4 + 1] = (uchar)((float)data_in[id * 4 + 1] * 255.0f / 65535.0f);
              data_out[id * 4 + 2] = (uchar)((float)data_in[id * 4 + 2] * 255.0f / 65535.0f);
              data_out[id * 4 + 3] = (uchar)((float)data_in[id * 4 + 3] * 255.0f / 65535.0f);
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_FLOAT4: {
          // return TextureInterpolator<float4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(uchar) * info.width * info.height * 4);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          uchar *data_out = (uchar *)mini_material->colorTexture->data.data();
          float *data_in = (float *)info.data;

          // memcpy(mini_material->colorTexture->data.data(),
          //       (char *)info.data,
          //       mini_material->colorTexture->data.size());
#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id * 4 + 0] = (uchar)(data_in[id * 4 + 0] * 255.0f);
              data_out[id * 4 + 1] = (uchar)(data_in[id * 4 + 1] * 255.0f);
              data_out[id * 4 + 2] = (uchar)(data_in[id * 4 + 2] * 255.0f);
              data_out[id * 4 + 3] = (uchar)(data_in[id * 4 + 3] * 255.0f);
            }
          }

          break;
        }
      }
    }
  }
  return offset;
}

int svm_node_tex_image(ccl::KernelGlobalsCPU *kg,
                       float *stack,
                       // ccl::uint4 node,
                       int num_nodes,
                       int offset,
                       mini::Material::SP mini_material)
{
  const char *byte4_force = getenv("CLIENT_FILE_TEXTURE_BYTE4_FORCE");
  if (byte4_force) {
    return svm_node_tex_image_byte4(kg,
                                    stack,
                                    // ccl::uint4 node,
                                    num_nodes,
                                    offset,
                                    mini_material);
  }
  // offset = svm_node_tex_image(kg, sd, stack, node, offset);
  // int num_nodes = (int)node.y;
  if (num_nodes <= 0) {
    int id = -num_nodes;
    const ccl::TextureInfo &info = kernel_data_fetch(texture_info, id);
    if (info.data) {
      mini_material->colorTexture = std::make_shared<mini::Texture>();
      set_tex_interpolation(info, mini_material->colorTexture);

      switch (info.data_type) {
        case ccl::IMAGE_DATA_TYPE_HALF: {
          // const float f = TextureInterpolator<ccl::half, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT1;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          float *data_out = (float *)mini_material->colorTexture->data.data();
          ccl::half *data_in = (ccl::half *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id] = half_to_float(data_in[id]);
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_BYTE: {
          // const float f = TextureInterpolator<uchar, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT1;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          float *data_out = (float *)mini_material->colorTexture->data.data();
          uchar *data_in = (uchar *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id] = (float)data_in[id] / (float)255.0f;
            }
          }

          break;
        }
        case ccl::IMAGE_DATA_TYPE_USHORT: {
          // const float f = TextureInterpolator<uint16_t, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT1;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          float *data_out = (float *)mini_material->colorTexture->data.data();
          uint16_t *data_in = (uint16_t *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id] = (float)data_in[id] / (float)65535.0f;
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_FLOAT: {
          // const float f = TextureInterpolator<float, float>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT1;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          memcpy(mini_material->colorTexture->data.data(),
                 (char *)info.data,
                 mini_material->colorTexture->data.size());

          break;
        }
        case ccl::IMAGE_DATA_TYPE_HALF4: {
          // return TextureInterpolator<half4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * 4 * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          float *data_out = (float *)mini_material->colorTexture->data.data();
          ccl::half *data_in = (ccl::half *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id * 4 + 0] = half_to_float(data_in[id * 4 + 0]);
              data_out[id * 4 + 1] = half_to_float(data_in[id * 4 + 1]);
              data_out[id * 4 + 2] = half_to_float(data_in[id * 4 + 2]);
              data_out[id * 4 + 3] = half_to_float(data_in[id * 4 + 3]);
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_BYTE4: {
          // return TextureInterpolator<uchar4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(char) * 4 * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::BYTE4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          memcpy(mini_material->colorTexture->data.data(),
                 (char *)info.data,
                 mini_material->colorTexture->data.size());
          break;
        }
        case ccl::IMAGE_DATA_TYPE_USHORT4: {
          // return TextureInterpolator<ushort4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * 4 * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          float *data_out = (float *)mini_material->colorTexture->data.data();
          uint16_t *data_in = (uint16_t *)info.data;

#pragma omp parallel for
          for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
              size_t id = x + (size_t)y * info.width;
              data_out[id * 4 + 0] = (float)data_in[id * 4 + 0] / (float)65535.0f;
              data_out[id * 4 + 1] = (float)data_in[id * 4 + 1] / (float)65535.0f;
              data_out[id * 4 + 2] = (float)data_in[id * 4 + 2] / (float)65535.0f;
              data_out[id * 4 + 3] = (float)data_in[id * 4 + 3] / (float)65535.0f;
            }
          }
          break;
        }
        case ccl::IMAGE_DATA_TYPE_FLOAT4: {
          // return TextureInterpolator<float4>::interp(info, x, y);
          mini_material->colorTexture->data.resize(sizeof(float) * 4 * info.width * info.height);
          mini_material->colorTexture->format = mini::Texture::FLOAT4;
          mini_material->colorTexture->size.x = info.width;
          mini_material->colorTexture->size.y = info.height;

          memcpy(mini_material->colorTexture->data.data(),
                 (char *)info.data,
                 mini_material->colorTexture->data.size());

          break;
        }
      }
    }
  }
  return offset;
}

#define SVM_CASE(node) case ccl::node:

void svm_eval_nodes(ccl::KernelGlobalsCPU *kg, ccl::uint shader, mini::Material::SP mini_material)
{
  float stack[SVM_STACK_SIZE];

  // ccl::uint shader = kernel_data_fetch(tri_shader, prim);
  int offset = shader & ccl::SHADER_MASK;
  ccl::ShaderType type = ccl::SHADER_TYPE_SURFACE;
  ccl::float3 sd_svm_closure_weight = ccl::make_float3(0, 0, 0);

  while (1) {
    ccl::uint4 node = read_node(kg, &offset);

#ifdef _WIN32

#  define SHADER_NODE_TYPE(name) \
    if (node.x == ccl::name) { \
      printf("%s: %d,%d,%d,%d ; %f,%f,%f,%f\n", \
             #name, \
             node.x, \
             node.y, \
             node.z, \
             node.w, \
             ccl::__uint_as_float(node.x), \
             ccl::__uint_as_float(node.y), \
             ccl::__uint_as_float(node.z), \
             ccl::__uint_as_float(node.w)); \
    }
#  include "kernel/svm/node_types_template.h"

#endif

    switch (node.x) {
      SVM_CASE(NODE_END)
      return;
      SVM_CASE(NODE_SHADER_JUMP)
      {
        if (type == ccl::SHADER_TYPE_SURFACE)
          offset = node.y;
        else if (type == ccl::SHADER_TYPE_VOLUME)
          offset = node.z;
        else if (type == ccl::SHADER_TYPE_DISPLACEMENT)
          offset = node.w;
        else
          return;
        break;
      }
      SVM_CASE(NODE_CLOSURE_BSDF)
      {
        // offset = svm_node_closure_bsdf<node_feature_mask, type>(
        //    kg, sd, stack, node, path_flag, offset);
        // break;
        uint path_flag = 0;
        int sd_flag = kernel_data_fetch(shaders, (shader & ccl::SHADER_MASK)).flags;

        offset = svm_node_closure_bsdf(
            kg, stack, node, path_flag, offset, sd_flag, sd_svm_closure_weight, mini_material);
        break;
      }
      SVM_CASE(NODE_CLOSURE_EMISSION)
      {
        // IF_KERNEL_NODES_FEATURE(EMISSION)
        //{
        //  svm_node_closure_emission(sd, stack, node);
        //}
        break;
      }
      SVM_CASE(NODE_CLOSURE_BACKGROUND)
      {
        // IF_KERNEL_NODES_FEATURE(EMISSION)
        //{
        //  svm_node_closure_background(sd, stack, node);
        //}
        break;
      }
      SVM_CASE(NODE_CLOSURE_SET_WEIGHT)
      {
        // svm_node_closure_set_weight(sd, node.y, node.z, node.w);
        sd_svm_closure_weight = ccl::make_float3(ccl::__uint_as_float(node.y),
                                                 ccl::__uint_as_float(node.z),
                                                 ccl::__uint_as_float(node.w));
        break;
      }
      SVM_CASE(NODE_CLOSURE_WEIGHT)
      {
        // svm_node_closure_weight(sd, stack, node.y);
        sd_svm_closure_weight = stack_load_float3(stack, node.y);
        break;
      }
      SVM_CASE(NODE_EMISSION_WEIGHT)
      {
        // IF_KERNEL_NODES_FEATURE(EMISSION)
        //{
        //  svm_node_emission_weight(kg, sd, stack, node);
        //}
        break;
      }
      SVM_CASE(NODE_MIX_CLOSURE)
      {
        // svm_node_mix_closure(sd, stack, node);
        /* fetch weight from blend input, previous mix closures,
         * and write to stack to be used by closure nodes later */
        uint weight_offset, in_weight_offset, weight1_offset, weight2_offset;
        svm_unpack_node_uchar4(
            node.y, &weight_offset, &in_weight_offset, &weight1_offset, &weight2_offset);

        float weight = stack_load_float(stack, weight_offset);
        weight = ccl::saturatef(weight);

        float in_weight = (stack_valid(in_weight_offset)) ?
                              stack_load_float(stack, in_weight_offset) :
                              1.0f;

        if (stack_valid(weight1_offset))
          stack_store_float(stack, weight1_offset, in_weight * (1.0f - weight));
        if (stack_valid(weight2_offset))
          stack_store_float(stack, weight2_offset, in_weight * weight);

        break;
      }
      SVM_CASE(NODE_JUMP_IF_ZERO)
      {
        if (stack_load_float(stack, node.z) <= 0.0f)
          offset += node.y;
        break;
      }
      SVM_CASE(NODE_JUMP_IF_ONE)
      {
        if (stack_load_float(stack, node.z) >= 1.0f)
          offset += node.y;
        break;
      }
      SVM_CASE(NODE_GEOMETRY)
      {
        // svm_node_geometry(kg, sd, stack, node.y, node.z);
        break;
      }
      SVM_CASE(NODE_CONVERT)
      {
        // svm_node_convert(kg, sd, stack, node.y, node.z, node.w);
        break;
      }
      SVM_CASE(NODE_TEX_COORD)
      {
        // offset = svm_node_tex_coord(kg, sd, path_flag, stack, node, offset);
        break;
      }
      SVM_CASE(NODE_VALUE_F)
      {
        // svm_node_value_f(kg, sd, stack, node.y, node.z);
        stack[node.z] = ccl::__uint_as_float(node.y);
        break;
      }
      SVM_CASE(NODE_VALUE_V)
      {
        // offset = svm_node_value_v(kg, sd, stack, node.y, offset);
        ccl::uint4 node1 = read_node(kg, &offset);
        float *stack_a = stack + node.y;
        stack_a[0] = ccl::__uint_as_float(node1.y);
        stack_a[1] = ccl::__uint_as_float(node1.z);
        stack_a[2] = ccl::__uint_as_float(node1.w);
        break;
      }
      SVM_CASE(NODE_ATTR)
      {
        // svm_node_attr<node_feature_mask>(kg, sd, stack, node);
        break;
      }
      SVM_CASE(NODE_VERTEX_COLOR)
      {
        // svm_node_vertex_color(kg, sd, stack, node.y, node.z, node.w);
        break;
      }
      SVM_CASE(NODE_GEOMETRY_BUMP_DX)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  svm_node_geometry_bump_dx(kg, sd, stack, node.y, node.z);
        //}
        break;
      }
      SVM_CASE(NODE_GEOMETRY_BUMP_DY)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  svm_node_geometry_bump_dy(kg, sd, stack, node.y, node.z);
        //}
        break;
      }
      SVM_CASE(NODE_SET_DISPLACEMENT)
      {
        // svm_node_set_displacement<node_feature_mask>(kg, sd, stack, node.y);
        break;
      }
      SVM_CASE(NODE_DISPLACEMENT)
      {
        // svm_node_displacement<node_feature_mask>(kg, sd, stack, node);
        break;
      }
      SVM_CASE(NODE_VECTOR_DISPLACEMENT)
      {
        // offset = svm_node_vector_displacement<node_feature_mask>(kg, sd, stack, node, offset);
        break;
      }
      SVM_CASE(NODE_TEX_IMAGE)
      {
        // offset = svm_node_tex_image(kg, sd, stack, node, offset);
        offset = svm_node_tex_image(kg, stack, (int)node.y, offset, mini_material);
        break;
      }
      SVM_CASE(NODE_TEX_IMAGE_BOX)
      {
        // svm_node_tex_image_box(kg, sd, stack, node);
        break;
      }
      SVM_CASE(NODE_TEX_NOISE)
      {
        // offset = svm_node_tex_noise(kg, sd, stack, node.y, node.z, node.w, offset);
        break;
      }
      SVM_CASE(NODE_SET_BUMP)
      {
        // svm_node_set_bump<node_feature_mask>(kg, sd, stack, node);
        break;
      }
      SVM_CASE(NODE_ATTR_BUMP_DX)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  svm_node_attr_bump_dx(kg, sd, stack, node);
        //}
        break;
      }
      SVM_CASE(NODE_ATTR_BUMP_DY)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  svm_node_attr_bump_dy(kg, sd, stack, node);
        //}
        break;
      }
      SVM_CASE(NODE_VERTEX_COLOR_BUMP_DX)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  svm_node_vertex_color_bump_dx(kg, sd, stack, node.y, node.z, node.w);
        //}
        break;
      }
      SVM_CASE(NODE_VERTEX_COLOR_BUMP_DY)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  svm_node_vertex_color_bump_dy(kg, sd, stack, node.y, node.z, node.w);
        //}
        break;
      }
      SVM_CASE(NODE_TEX_COORD_BUMP_DX)
      {
        // IF_KERNEL_NODES_FEATURE(BUMP)
        //{
        //  offset = svm_node_tex_coord_bump_dx(kg, sd, path_flag, stack, node, offset);
        //}
        break;
      }
      //      SVM_CASE(NODE_TEX_COORD_BUMP_DY)
      //      IF_KERNEL_NODES_FEATURE(BUMP)
      //      {
      //        offset = svm_node_tex_coord_bump_dy(kg, sd, path_flag, stack, node, offset);
      //      }
      //      break;
      //      SVM_CASE(NODE_CLOSURE_SET_NORMAL)
      //      IF_KERNEL_NODES_FEATURE(BUMP)
      //      {
      //        svm_node_set_normal(kg, sd, stack, node.y, node.z);
      //      }
      //      break;
      //      SVM_CASE(NODE_ENTER_BUMP_EVAL)
      //      IF_KERNEL_NODES_FEATURE(BUMP_STATE)
      //      {
      //        svm_node_enter_bump_eval(kg, sd, stack, node.y);
      //      }
      //      break;
      //      SVM_CASE(NODE_LEAVE_BUMP_EVAL)
      //      IF_KERNEL_NODES_FEATURE(BUMP_STATE)
      //      {
      //        svm_node_leave_bump_eval(kg, sd, stack, node.y);
      //      }
      //      break;
      //      SVM_CASE(NODE_HSV)
      //      svm_node_hsv(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_CLOSURE_HOLDOUT)
      //      svm_node_closure_holdout(sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_FRESNEL)
      //      svm_node_fresnel(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_LAYER_WEIGHT)
      //      svm_node_layer_weight(sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_CLOSURE_VOLUME)
      //      IF_KERNEL_NODES_FEATURE(VOLUME)
      //      {
      //        svm_node_closure_volume<type>(kg, sd, stack, node);
      //      }
      //      break;
      //      SVM_CASE(NODE_PRINCIPLED_VOLUME)
      //      IF_KERNEL_NODES_FEATURE(VOLUME)
      //      {
      //        offset = svm_node_principled_volume<type>(kg, sd, stack, node, path_flag, offset);
      //      }
      //      break;
      //      SVM_CASE(NODE_MATH)
      //      svm_node_math(kg, sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_VECTOR_MATH)
      //      offset = svm_node_vector_math(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_RGB_RAMP)
      //      offset = svm_node_rgb_ramp(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_GAMMA)
      //      svm_node_gamma(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_BRIGHTCONTRAST)
      //      svm_node_brightness(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_LIGHT_PATH)
      //      svm_node_light_path<node_feature_mask>(kg, state, sd, stack, node.y, node.z,
      //      path_flag); break; SVM_CASE(NODE_OBJECT_INFO) svm_node_object_info(kg, sd, stack,
      //      node.y, node.z); break; SVM_CASE(NODE_PARTICLE_INFO) svm_node_particle_info(kg, sd,
      //      stack, node.y, node.z); break;
      //#if defined(__HAIR__)
      //      SVM_CASE(NODE_HAIR_INFO)
      //      svm_node_hair_info(kg, sd, stack, node.y, node.z);
      //      break;
      //#endif
      //#if defined(__POINTCLOUD__)
      //      SVM_CASE(NODE_POINT_INFO)
      //      svm_node_point_info(kg, sd, stack, node.y, node.z);
      //      break;
      //#endif
      //      SVM_CASE(NODE_TEXTURE_MAPPING)
      //      offset = svm_node_texture_mapping(kg, sd, stack, node.y, node.z, offset);
      //      break;
      //      SVM_CASE(NODE_MAPPING)
      //      svm_node_mapping(kg, sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_MIN_MAX)
      //      offset = svm_node_min_max(kg, sd, stack, node.y, node.z, offset);
      //      break;
      //      SVM_CASE(NODE_CAMERA)
      //      svm_node_camera(kg, sd, stack, node.y, node.z, node.w);
      //      break;
      SVM_CASE(NODE_TEX_ENVIRONMENT)
      {
        // svm_node_tex_environment(kg, sd, stack, node);
        uint id = node.y;
        uint co_offset, out_offset, alpha_offset, flags;
        uint projection = node.w;

        svm_unpack_node_uchar4(node.z, &co_offset, &out_offset, &alpha_offset, &flags);

        ccl::float3 co = stack_load_float3(stack, co_offset);
        ccl::float2 uv;

        // co = ccl::safe_normalize(co);

        // if (projection == 0)
        //  uv = direction_to_equirectangular(co);
        // else
        //  uv = direction_to_mirrorball(co);

        // float4 f = svm_image_texture(kg, id, uv.x, uv.y, flags);
        svm_node_tex_image(kg, stack, -id, offset, mini_material);

        // if (stack_valid(out_offset))
        //  stack_store_float3(stack, out_offset, make_float3(f.x, f.y, f.z));
        // if (stack_valid(alpha_offset))
        //  stack_store_float(stack, alpha_offset, f.w);
        break;
      }
      //      SVM_CASE(NODE_TEX_SKY)
      //      offset = svm_node_tex_sky(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_TEX_GRADIENT)
      //      svm_node_tex_gradient(sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_TEX_VORONOI)
      //      offset = svm_node_tex_voronoi<node_feature_mask>(
      //          kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_TEX_MUSGRAVE)
      //      offset = svm_node_tex_musgrave(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_TEX_WAVE)
      //      offset = svm_node_tex_wave(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_TEX_MAGIC)
      //      offset = svm_node_tex_magic(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_TEX_CHECKER)
      //      svm_node_tex_checker(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_TEX_BRICK)
      //      offset = svm_node_tex_brick(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_TEX_WHITE_NOISE)
      //      svm_node_tex_white_noise(kg, sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_NORMAL)
      //      offset = svm_node_normal(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_LIGHT_FALLOFF)
      //      svm_node_light_falloff(sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_IES)
      //      svm_node_ies(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_CURVES)
      //      offset = svm_node_curves(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_FLOAT_CURVE)
      //      offset = svm_node_curve(kg, sd, stack, node, offset);
      //      break;
      //      SVM_CASE(NODE_TANGENT)
      //      svm_node_tangent(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_NORMAL_MAP)
      //      svm_node_normal_map(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_INVERT)
      //      svm_node_invert(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_MIX)
      //      offset = svm_node_mix(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_SEPARATE_COLOR)
      //      svm_node_separate_color(kg, sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_COMBINE_COLOR)
      //      svm_node_combine_color(kg, sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_SEPARATE_VECTOR)
      //      svm_node_separate_vector(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_COMBINE_VECTOR)
      //      svm_node_combine_vector(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_SEPARATE_HSV)
      //      offset = svm_node_separate_hsv(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_COMBINE_HSV)
      //      offset = svm_node_combine_hsv(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_VECTOR_ROTATE)
      //      svm_node_vector_rotate(sd, stack, node.y, node.z, node.w);
      //      break;
      //      SVM_CASE(NODE_VECTOR_TRANSFORM)
      //      svm_node_vector_transform(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_WIREFRAME)
      //      svm_node_wireframe(kg, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_WAVELENGTH)
      //      svm_node_wavelength(kg, sd, stack, node.y, node.z);
      //      break;
      //      SVM_CASE(NODE_BLACKBODY)
      //      svm_node_blackbody(kg, sd, stack, node.y, node.z);
      //      break;
      //      SVM_CASE(NODE_MAP_RANGE)
      //      offset = svm_node_map_range(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_VECTOR_MAP_RANGE)
      //      offset = svm_node_vector_map_range(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //      SVM_CASE(NODE_CLAMP)
      //      offset = svm_node_clamp(kg, sd, stack, node.y, node.z, node.w, offset);
      //      break;
      //#ifdef __SHADER_RAYTRACE__
      //      SVM_CASE(NODE_BEVEL)
      //      svm_node_bevel<node_feature_mask>(kg, state, sd, stack, node);
      //      break;
      //      SVM_CASE(NODE_AMBIENT_OCCLUSION)
      //      svm_node_ao<node_feature_mask>(kg, state, sd, stack, node);
      //      break;
      //#endif
      //
      //      SVM_CASE(NODE_TEX_VOXEL)
      //      IF_KERNEL_NODES_FEATURE(VOLUME)
      //      {
      //        offset = svm_node_tex_voxel(kg, sd, stack, node, offset);
      //      }
      //      break;
      //      SVM_CASE(NODE_AOV_START)
      //      if (!svm_node_aov_check(path_flag, render_buffer)) {
      //        return;
      //      }
      //      break;
      //      SVM_CASE(NODE_AOV_COLOR)
      //      svm_node_aov_color<node_feature_mask>(kg, state, sd, stack, node, render_buffer);
      //      break;
      //      SVM_CASE(NODE_AOV_VALUE)
      //      svm_node_aov_value<node_feature_mask>(kg, state, sd, stack, node, render_buffer);
      //      break;
      //      default:
      //        kernel_assert(!"Unknown node type was passed to the SVM machine");
      //        return;
    }
  }
}

////////////
uint subd_triangle_patch(ccl::KernelGlobals kg, uint prim)
{
  return (prim != PRIM_NONE) ? kernel_data_fetch(tri_patch, prim) : ~0;
}

uint attribute_primitive_type(ccl::KernelGlobals kg, uint prim)
{
  if (subd_triangle_patch(kg, prim) != ~0) {
    return ccl::ATTR_PRIM_SUBD;
  }
  else {
    return ccl::ATTR_PRIM_GEOMETRY;
  }
}
ccl::AttributeDescriptor attribute_not_found()
{
  const ccl::AttributeDescriptor desc = {
      ccl::ATTR_ELEMENT_NONE, (ccl::NodeAttributeType)0, 0, ccl::ATTR_STD_NOT_FOUND};
  return desc;
}

/* Find attribute based on ID */

uint object_attribute_map_offset(ccl::KernelGlobalsCPU *kg, int object)
{
  return kernel_data_fetch(objects, object).attribute_map_offset;
}

ccl::AttributeDescriptor find_attribute(ccl::KernelGlobalsCPU *kg,
                                        uint sd_object,
                                        uint prim,
                                        uint id)
{
  if (sd_object == OBJECT_NONE) {
    return attribute_not_found();
  }

  /* for SVM, find attribute by unique id */
  uint attr_offset = object_attribute_map_offset(kg, sd_object);
  attr_offset += attribute_primitive_type(kg, prim);
  ccl::AttributeMap attr_map = kernel_data_fetch(attributes_map, attr_offset);

  while (attr_map.id != id) {
    if (attr_map.id == ccl::ATTR_STD_NONE) {
      if (attr_map.element == 0) {
        return attribute_not_found();
      }
      else {
        /* Chain jump to a different part of the table. */
        attr_offset = attr_map.offset;
      }
    }
    else {
      attr_offset += ccl::ATTR_PRIM_TYPES;
    }
    attr_map = kernel_data_fetch(attributes_map, attr_offset);
  }

  ccl::AttributeDescriptor desc;
  desc.element = (ccl::AttributeElement)attr_map.element;

  if (prim == PRIM_NONE && desc.element != ccl::ATTR_ELEMENT_MESH &&
      desc.element != ccl::ATTR_ELEMENT_VOXEL && desc.element != ccl::ATTR_ELEMENT_OBJECT) {
    return attribute_not_found();
  }

  /* return result */
  desc.offset = (attr_map.element == ccl::ATTR_ELEMENT_NONE) ? (int)ccl::ATTR_STD_NOT_FOUND :
                                                               (int)attr_map.offset;
  desc.type = (ccl::NodeAttributeType)attr_map.type;
  desc.flags = (ccl::AttributeFlag)attr_map.flags;

  return desc;
}
////////////

void kg_to_mini(ccl::KernelGlobalsCPU *kg, const char *filename)
{
  ////////////////////////////////////////////////////////////

  ////BVH2, not used for OptiX or Embree.
  // KERNEL_DATA_ARRAY(float4, bvh_nodes)
  // KERNEL_DATA_ARRAY(float4, bvh_leaf_nodes)
  // KERNEL_DATA_ARRAY(uint, prim_type)
  // KERNEL_DATA_ARRAY(uint, prim_visibility)
  // KERNEL_DATA_ARRAY(uint, prim_index)
  // KERNEL_DATA_ARRAY(uint, prim_object)
  // KERNEL_DATA_ARRAY(uint, object_node)
  // KERNEL_DATA_ARRAY(float2, prim_time)

  //// objects
  // KERNEL_DATA_ARRAY(KernelObject, objects)
  // KERNEL_DATA_ARRAY(Transform, object_motion_pass)
  // KERNEL_DATA_ARRAY(DecomposedTransform, object_motion)
  // KERNEL_DATA_ARRAY(uint, object_flag)
  // KERNEL_DATA_ARRAY(float, object_volume_step)
  // KERNEL_DATA_ARRAY(uint, object_prim_offset)
  // KERNEL_DATA_ARRAY(uint, object_prim_count)

  //// cameras
  // KERNEL_DATA_ARRAY(DecomposedTransform, camera_motion)

  //// triangles
  // KERNEL_DATA_ARRAY(uint, tri_shader)
  // KERNEL_DATA_ARRAY(packed_float3, tri_vnormal)
  // KERNEL_DATA_ARRAY(uint4, tri_vindex)
  // KERNEL_DATA_ARRAY(uint, tri_patch)
  // KERNEL_DATA_ARRAY(float2, tri_patch_uv)
  // KERNEL_DATA_ARRAY(packed_float3, tri_verts)

  //// curves
  // KERNEL_DATA_ARRAY(KernelCurve, curves)
  // KERNEL_DATA_ARRAY(float4, curve_keys)
  // KERNEL_DATA_ARRAY(KernelCurveSegment, curve_segments)

  //// patches
  // KERNEL_DATA_ARRAY(uint, patches)

  //// pointclouds
  // KERNEL_DATA_ARRAY(float4, points)
  // KERNEL_DATA_ARRAY(uint, points_shader)

  //// attributes
  // KERNEL_DATA_ARRAY(AttributeMap, attributes_map)
  // KERNEL_DATA_ARRAY(float, attributes_float)
  // KERNEL_DATA_ARRAY(float2, attributes_float2)
  // KERNEL_DATA_ARRAY(packed_float3, attributes_float3)
  // KERNEL_DATA_ARRAY(float4, attributes_float4)
  // KERNEL_DATA_ARRAY(uchar4, attributes_uchar4)

  //// lights
  // KERNEL_DATA_ARRAY(KernelLightDistribution, light_distribution)
  // KERNEL_DATA_ARRAY(KernelLight, lights)
  // KERNEL_DATA_ARRAY(float2, light_background_marginal_cdf)
  // KERNEL_DATA_ARRAY(float2, light_background_conditional_cdf)

  //// particles
  // KERNEL_DATA_ARRAY(KernelParticle, particles)

  //// shaders
  // KERNEL_DATA_ARRAY(uint4, svm_nodes)
  // KERNEL_DATA_ARRAY(KernelShader, shaders)

  //// lookup tables
  // KERNEL_DATA_ARRAY(float, lookup_table)

  //// sobol
  // KERNEL_DATA_ARRAY(float, sample_pattern_lut)

  //// image textures
  // KERNEL_DATA_ARRAY(TextureInfo, texture_info)

  //// ies lights
  // KERNEL_DATA_ARRAY(float, ies)

  //////////////////////////////////////////////
  mini::Scene::SP mini_scene = std::make_shared<mini::Scene>();

  /////////////////////LIGHT/////////////////////////
  //  LIGHT_POINT, LIGHT_DISTANT, LIGHT_BACKGROUND,
  size_t light_size = kg->lights.width;
  for (int i = 0; i < light_size; i++) {
    ccl::KernelLight klight = kernel_data_fetch(lights, i);
    switch (klight.type) {
      case (ccl::LIGHT_DISTANT): {
        mini::DirLight mini_light;

        ccl::float3 dir_light = ccl::transform_direction(&klight.tfm, ccl::make_float3(0, 0, -1));

        mini_light.direction[0] = dir_light[0];
        mini_light.direction[1] = dir_light[1];
        mini_light.direction[2] = dir_light[2];

        mini_light.radiance[0] = klight.strength[0];
        mini_light.radiance[1] = klight.strength[1];
        mini_light.radiance[2] = klight.strength[2];

        mini_scene->dirLights.push_back(mini_light);

        break;
      }
      case (ccl::LIGHT_BACKGROUND): {
        mini_scene->envMapLight = std::make_shared<mini::EnvMapLight>();
        mini_scene->envMapLight->texture = std::make_shared<mini::Texture>();

        // kernel_data.background.surface_shader
        mini::Material::SP mat = std::make_shared<mini::Material>();
        svm_eval_nodes(kg, klight.shader_id, mat);

        mini_scene->envMapLight->texture = mat->colorTexture;

        mini_scene->envMapLight->transform.l.vx.x = -klight.tfm.x.x;
        mini_scene->envMapLight->transform.l.vy.x = -klight.tfm.x.y;
        mini_scene->envMapLight->transform.l.vz.x = -klight.tfm.x.z;

        mini_scene->envMapLight->transform.l.vx.y = klight.tfm.y.x;
        mini_scene->envMapLight->transform.l.vy.y = klight.tfm.y.y;
        mini_scene->envMapLight->transform.l.vz.y = klight.tfm.y.z;

        mini_scene->envMapLight->transform.l.vx.z = -klight.tfm.z.x;
        mini_scene->envMapLight->transform.l.vy.z = -klight.tfm.z.y;
        mini_scene->envMapLight->transform.l.vz.z = -klight.tfm.z.z;

        mini_scene->envMapLight->transform.p.x = klight.tfm.x.w;
        mini_scene->envMapLight->transform.p.y = klight.tfm.y.w;
        mini_scene->envMapLight->transform.p.z = klight.tfm.z.w;

        // mini_scene->envMapLight->transform.l.vx.x = klight.itfm.x.x;
        // mini_scene->envMapLight->transform.l.vy.x = klight.itfm.x.y;
        // mini_scene->envMapLight->transform.l.vz.x = klight.itfm.x.z;

        // mini_scene->envMapLight->transform.l.vx.y = klight.itfm.y.x;
        // mini_scene->envMapLight->transform.l.vy.y = klight.itfm.y.y;
        // mini_scene->envMapLight->transform.l.vz.y = klight.itfm.y.z;

        // mini_scene->envMapLight->transform.l.vx.z = klight.itfm.z.x;
        // mini_scene->envMapLight->transform.l.vy.z = klight.itfm.z.y;
        // mini_scene->envMapLight->transform.l.vz.z = klight.itfm.z.z;

        // mini_scene->envMapLight->transform.p.x = klight.itfm.x.w;
        // mini_scene->envMapLight->transform.p.y = klight.itfm.y.w;
        // mini_scene->envMapLight->transform.p.z = klight.itfm.z.w;
        break;
      }
    }
  }
  /////////////////////MESH/////////////////////////
  size_t obj_size = (kg->data.bvh.root != -1) ? kg->objects.width : 0;
  size_t vindex_size = (kg->data.bvh.root != -1) ? kg->tri_vindex.width : 0;
  size_t verts_size = (kg->data.bvh.root != -1) ? kg->tri_verts.width : 0;

  size_t mini_obj_size = 0;
  size_t mini_vindex_size = 0;
  size_t mini_verts_size = 0;

#if 0
  struct temp_scene {
    ccl::uint prim_offset;
    ccl::uint prim_count;
    ccl::uint start_obj;
    ccl::uint end_obj;
  };

  std::vector<temp_scene> temp_scenes;
#endif
  struct temp_shader {
    std::unordered_map<ccl::uint, ccl::uint> obj;
    ccl::uint shader;

    ccl::uint prim_offset;
    ccl::uint prim_count;

    std::unordered_map<ccl::uint, ccl::uint> prim;
  };

  std::unordered_map<ccl::uint,
                     std::unordered_map<ccl::uint, std::unordered_map<ccl::uint, temp_shader>>>
      temp_shaders;

  // temp_scene ts;
  // ts.start_prim = kernel_data_fetch(object_prim_offset, 0);
  // ts.end_prim = ts.start_prim + kernel_data_fetch(object_prim_count, 0);
  // ts.start_obj = 0;
  // ts.end_obj = 0;
  // temp_scenes.push_back(ts);

#if 0
  for (size_t i = 0; i < obj_size; i++) {
    ccl::uint prim_offset = kernel_data_fetch(object_prim_offset, i);
    ccl::uint prim_count = kernel_data_fetch(object_prim_count, i);

    if (prim_count == 0)
      continue;

    if (temp_scenes.size() == 0 ||
        prim_offset != temp_scenes[temp_scenes.size() - 1].prim_offset ||
        prim_count != temp_scenes[temp_scenes.size() - 1].prim_count) {

      ccl::uint shader_prev = kernel_data_fetch(tri_shader, prim_offset);

      ccl::uint prim_offset2 = prim_offset;

      //for (ccl::uint prim = prim_offset + 1; prim < prim_offset + prim_count; prim++) {
      //  ccl::uint shader = kernel_data_fetch(tri_shader, prim);

      //  if (shader_prev != shader) {
      //    temp_scene ts;
      //    ts.prim_offset = prim_offset2;
      //    ts.prim_count = prim - prim_offset2;
      //    ts.start_obj = i;
      //    ts.end_obj = i;
      //    temp_scenes.push_back(ts);

      //    prim_offset2 = prim;
      //    shader_prev = shader;
      //  }
      //}

      if (prim_count - (prim_offset2 - prim_offset) > 0) {
        temp_scene ts;
        ts.prim_offset = prim_offset2;
        ts.prim_count = prim_count - (prim_offset2 - prim_offset);
        ts.start_obj = i;
        ts.end_obj = i;
        temp_scenes.push_back(ts);
      }

      continue;
    }

    temp_scenes[temp_scenes.size() - 1].end_obj++;
  }
#endif
  ///////////////////////////////////////////////////////////////////////////////////////////////
  for (size_t i = 0; i < obj_size; i++) {
    ccl::uint prim_offset = kernel_data_fetch(object_prim_offset, i);
    ccl::uint prim_count = kernel_data_fetch(object_prim_count, i);

    for (ccl::uint prim = prim_offset; prim < prim_offset + prim_count; prim++) {
      ccl::uint shader = kernel_data_fetch(tri_shader, prim);
      temp_shaders[shader][prim_offset][prim_count].prim_offset = prim_offset;
      temp_shaders[shader][prim_offset][prim_count].prim_count = prim_count;
      temp_shaders[shader][prim_offset][prim_count].obj[i] = i;
      temp_shaders[shader][prim_offset][prim_count].shader = shader;
      temp_shaders[shader][prim_offset][prim_count].prim[prim] = prim;
    }
  }

#if 1
  // temp_shaders[shader][prim_offset][prim_count]
  std::unordered_map<ccl::uint,
                     std::unordered_map<ccl::uint, std::unordered_map<ccl::uint, temp_shader>>>::
      iterator shader_it;

  std::unordered_map<ccl::uint, std::unordered_map<ccl::uint, temp_shader>>::iterator
      prim_offset_it;

  std::unordered_map<ccl::uint, temp_shader>::iterator prim_count_it;

  // shader_it
  for (shader_it = temp_shaders.begin(); shader_it != temp_shaders.end(); shader_it++) {
    // prim_offset_it
    for (prim_offset_it = shader_it->second.begin(); prim_offset_it != shader_it->second.end();
         prim_offset_it++) {
      // prim_count_it
      for (prim_count_it = prim_offset_it->second.begin();
           prim_count_it != prim_offset_it->second.end();
           prim_count_it++) {

        temp_shader &ts = prim_count_it->second;
        std::unordered_map<ccl::uint, ccl::uint>::iterator obj_it;
        std::unordered_map<ccl::uint, ccl::uint>::iterator prim_it;

        mini::Mesh::SP mini_mesh = std::make_shared<mini::Mesh>();

        // temp_shaders_it->second->;

        // svm_eval_nodes(kg, kernel_data_fetch(tri_shader, ts.prim_offset), mini_mesh->material);
        svm_eval_nodes(
            kg, kernel_data_fetch(tri_shader, ts.prim.begin()->second), mini_mesh->material);

        // uint attribute_map_offset =
        //    kernel_data_fetch(objects, ts.start_obj).attribute_map_offset;

        // for (ccl::uint prim = ts.prim_offset; prim < ts.prim_offset + ts.prim_count; prim++) {
        // for (ccl::uint p = 0; p < ts.prim.size(); p++) {
        for (prim_it = ts.prim.begin(); prim_it != ts.prim.end(); prim_it++) {
          ccl::uint prim = prim_it->second;
          // tri
          ccl::uint4 tri_vindex = kernel_data_fetch(tri_vindex, prim);

          mini::common::vec3i mini_index;
          mini_index[0] = mini_mesh->vertices.size();
          mini_index[1] = mini_index[0] + 1;
          mini_index[2] = mini_index[0] + 2;

          mini_mesh->indices.push_back(mini_index);

          // verts
          ccl::float3 tri0 = kernel_data_fetch(tri_verts, tri_vindex.w + 0);
          ccl::float3 tri1 = kernel_data_fetch(tri_verts, tri_vindex.w + 1);
          ccl::float3 tri2 = kernel_data_fetch(tri_verts, tri_vindex.w + 2);

          mini::common::vec3f mini_tri0;
          memcpy(&mini_tri0[0], &tri0[0], sizeof(float) * 3);
          mini::common::vec3f mini_tri1;
          memcpy(&mini_tri1[0], &tri1[0], sizeof(float) * 3);
          mini::common::vec3f mini_tri2;
          memcpy(&mini_tri2[0], &tri2[0], sizeof(float) * 3);

          mini_mesh->vertices.push_back(mini_tri0);
          mini_mesh->vertices.push_back(mini_tri1);
          mini_mesh->vertices.push_back(mini_tri2);

          // norms
          ccl::float3 n0 = kernel_data_fetch(tri_vnormal, tri_vindex.x);
          ccl::float3 n1 = kernel_data_fetch(tri_vnormal, tri_vindex.y);
          ccl::float3 n2 = kernel_data_fetch(tri_vnormal, tri_vindex.z);

          mini::common::vec3f mini_n0;
          memcpy(&mini_n0[0], &n0[0], sizeof(float) * 3);
          mini::common::vec3f mini_n1;
          memcpy(&mini_n1[0], &n1[0], sizeof(float) * 3);
          mini::common::vec3f mini_n2;
          memcpy(&mini_n2[0], &n2[0], sizeof(float) * 3);

          mini_mesh->normals.push_back(mini_n0);
          mini_mesh->normals.push_back(mini_n1);
          mini_mesh->normals.push_back(mini_n2);

          // texcoord
          if (mini_mesh->material->colorTexture) {
            // uint attribute_map_offset = kernel_data_fetch(objects,
            // ts.obj[0]).attribute_map_offset;

            // ccl::float2 attributes_float2 = kernel_data_fetch(attributes_float2,
            //                                                  attribute_map_offset + prim);

            ccl::AttributeDescriptor desc;

            for (obj_it = ts.obj.begin(); obj_it != ts.obj.end(); obj_it++) {
              desc = find_attribute(kg, obj_it->second, prim, ccl::ATTR_STD_NUM);
              if (desc.element & (ccl::ATTR_ELEMENT_VERTEX | ccl::ATTR_ELEMENT_VERTEX_MOTION |
                                  ccl::ATTR_ELEMENT_CORNER)) {
                break;
              }

              desc = find_attribute(kg, obj_it->second, prim, ccl::ATTR_STD_UV);
              if (desc.element & (ccl::ATTR_ELEMENT_VERTEX | ccl::ATTR_ELEMENT_VERTEX_MOTION |
                                  ccl::ATTR_ELEMENT_CORNER)) {
                break;
              }
            }

            if (desc.offset == ccl::ATTR_STD_NOT_FOUND || desc.element == ccl::ATTR_ELEMENT_NONE) {
              mini::common::vec2f tex_empty;
              memset(&tex_empty[0], 0, sizeof(float) * 2);

              mini_mesh->texcoords.push_back(tex_empty);
              mini_mesh->texcoords.push_back(tex_empty);
              mini_mesh->texcoords.push_back(tex_empty);
            }
            else {
              ccl::float2 attr0, attr1, attr2;

              if (desc.element & (ccl::ATTR_ELEMENT_VERTEX | ccl::ATTR_ELEMENT_VERTEX_MOTION |
                                  ccl::ATTR_ELEMENT_CORNER)) {

                if (desc.element & (ccl::ATTR_ELEMENT_VERTEX | ccl::ATTR_ELEMENT_VERTEX_MOTION)) {
                  attr0 = kernel_data_fetch(attributes_float2, desc.offset + tri_vindex.x);
                  attr1 = kernel_data_fetch(attributes_float2, desc.offset + tri_vindex.y);
                  attr2 = kernel_data_fetch(attributes_float2, desc.offset + tri_vindex.z);
                }
                else {
                  const int tri = desc.offset + prim * 3;
                  attr0 = kernel_data_fetch(attributes_float2, tri + 0);
                  attr1 = kernel_data_fetch(attributes_float2, tri + 1);
                  attr2 = kernel_data_fetch(attributes_float2, tri + 2);
                }
              }
              else {
                if (desc.element &
                    (ccl::ATTR_ELEMENT_FACE | ccl::ATTR_ELEMENT_OBJECT | ccl::ATTR_ELEMENT_MESH)) {
                  const int offset = (desc.element == ccl::ATTR_ELEMENT_FACE) ?
                                         desc.offset + prim :
                                         desc.offset;

                  attr0 = kernel_data_fetch(attributes_float2, offset);
                  attr1 = kernel_data_fetch(attributes_float2, offset);
                  attr2 = kernel_data_fetch(attributes_float2, offset);
                }
                else {
                  attr0 = ccl::make_float2(0.0f, 0.0f);
                  attr1 = ccl::make_float2(0.0f, 0.0f);
                  attr2 = ccl::make_float2(0.0f, 0.0f);
                }
              }

              mini::common::vec2f tex0;
              tex0[0] = attr0[0];
              tex0[1] = attr0[1];
              mini_mesh->texcoords.push_back(tex0);

              mini::common::vec2f tex1;
              tex1[0] = attr1[0];
              tex1[1] = attr1[1];
              mini_mesh->texcoords.push_back(tex1);

              mini::common::vec2f tex2;
              tex2[0] = attr2[0];
              tex2[1] = attr2[1];
              mini_mesh->texcoords.push_back(tex2);
            }
          }
          else {
            mini::common::vec2f tex_empty;
            memset(&tex_empty[0], 0, sizeof(float) * 2);

            mini_mesh->texcoords.push_back(tex_empty);
            mini_mesh->texcoords.push_back(tex_empty);
            mini_mesh->texcoords.push_back(tex_empty);
          }
        }

        if (mini_mesh->indices.size() == 0)
          continue;

        mini_obj_size++;
        mini_vindex_size += mini_mesh->indices.size();
        mini_verts_size += mini_mesh->vertices.size();

        mini::Object::SP mini_object = std::make_shared<mini::Object>();
        mini_object->meshes.push_back(mini_mesh);

        for (obj_it = ts.obj.begin(); obj_it != ts.obj.end(); obj_it++) {
          ccl::uint obj = obj_it->second;
          ccl::KernelObject kobject = kernel_data_fetch(objects, obj);
          ccl::uint object_flag = kernel_data_fetch(object_flag, obj);

          mini::Instance::SP mini_instance = std::make_shared<mini::Instance>(mini_object);

          if (!(object_flag & ccl::SD_OBJECT_TRANSFORM_APPLIED)) {
            mini_instance->xfm.l.vx.x = kobject.tfm.x.x;
            mini_instance->xfm.l.vy.x = kobject.tfm.x.y;
            mini_instance->xfm.l.vz.x = kobject.tfm.x.z;

            mini_instance->xfm.l.vx.y = kobject.tfm.y.x;
            mini_instance->xfm.l.vy.y = kobject.tfm.y.y;
            mini_instance->xfm.l.vz.y = kobject.tfm.y.z;

            mini_instance->xfm.l.vx.z = kobject.tfm.z.x;
            mini_instance->xfm.l.vy.z = kobject.tfm.z.y;
            mini_instance->xfm.l.vz.z = kobject.tfm.z.z;

            mini_instance->xfm.p.x = kobject.tfm.x.w;
            mini_instance->xfm.p.y = kobject.tfm.y.w;
            mini_instance->xfm.p.z = kobject.tfm.z.w;
          }

          mini_scene->instances.push_back(mini_instance);
        }
      }
    }
  }
#endif
  ///////////////////////////////////////////////////////////////////////////////////////////////

  // temp_scenes[temp_scenes.size() - 1].end_prim = vindex_size - 1;
  // temp_scenes[temp_scenes.size() - 1].end_prim = temp_scenes[temp_scenes.size() - 1].start_prim
  // +
  //                                               kernel_data_fetch(objects, obj_size -
  //                                               1).numverts / 3;

  /////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////

  // ccl::uint prim_last = kernel_data_fetch(object_prim_offset, 0);

  // int switch_test = 0;

#if 0
  for (size_t i = 0; i < temp_scenes.size(); i++) {
    mini::Mesh::SP mini_mesh = std::make_shared<mini::Mesh>();

    svm_eval_nodes(kg, temp_scenes[i].prim_offset, mini_mesh->material);
    uint attribute_map_offset =
        kernel_data_fetch(objects, temp_scenes[i].start_obj).attribute_map_offset;

    for (ccl::uint prim = temp_scenes[i].prim_offset;
         prim < temp_scenes[i].prim_offset + temp_scenes[i].prim_count;
         prim++) {
      // tri
      ccl::uint4 tri_vindex = kernel_data_fetch(tri_vindex, prim);

      mini::common::vec3i mini_index;
      mini_index[0] = mini_mesh->vertices.size();
      mini_index[1] = mini_index[0] + 1;
      mini_index[2] = mini_index[0] + 2;

      mini_mesh->indices.push_back(mini_index);

      // verts
      ccl::float3 tri0 = kernel_data_fetch(tri_verts, tri_vindex.w + 0);
      ccl::float3 tri1 = kernel_data_fetch(tri_verts, tri_vindex.w + 1);
      ccl::float3 tri2 = kernel_data_fetch(tri_verts, tri_vindex.w + 2);

      mini::common::vec3f mini_tri0;
      memcpy(&mini_tri0[0], &tri0[0], sizeof(float) * 3);
      mini::common::vec3f mini_tri1;
      memcpy(&mini_tri1[0], &tri1[0], sizeof(float) * 3);
      mini::common::vec3f mini_tri2;
      memcpy(&mini_tri2[0], &tri2[0], sizeof(float) * 3);

      mini_mesh->vertices.push_back(mini_tri0);
      mini_mesh->vertices.push_back(mini_tri1);
      mini_mesh->vertices.push_back(mini_tri2);

      // norms
      ccl::float3 n0 = kernel_data_fetch(tri_vnormal, tri_vindex.x);
      ccl::float3 n1 = kernel_data_fetch(tri_vnormal, tri_vindex.y);
      ccl::float3 n2 = kernel_data_fetch(tri_vnormal, tri_vindex.z);

      mini::common::vec3f mini_n0;
      memcpy(&mini_n0[0], &n0[0], sizeof(float) * 3);
      mini::common::vec3f mini_n1;
      memcpy(&mini_n1[0], &n1[0], sizeof(float) * 3);
      mini::common::vec3f mini_n2;
      memcpy(&mini_n2[0], &n2[0], sizeof(float) * 3);

      mini_mesh->normals.push_back(mini_n0);
      mini_mesh->normals.push_back(mini_n1);
      mini_mesh->normals.push_back(mini_n2);

      // texcoord
      if (mini_mesh->material->colorTexture) {
        // ccl::float2 attributes_float2 = kernel_data_fetch(
        //    attributes_float2, attribute_map_offset + (prim - temp_scenes[i].prim_offset));
        const ccl::AttributeDescriptor desc = find_attribute(
            kg, temp_scenes[i].start_obj, prim, ccl::ATTR_STD_UV);

        if (desc.element == ccl::ATTR_ELEMENT_NONE) {
          mini::common::vec2f tex_empty;
          memset(&tex_empty[0], 0, sizeof(float) * 2);

          mini_mesh->texcoords.push_back(tex_empty);
          mini_mesh->texcoords.push_back(tex_empty);
          mini_mesh->texcoords.push_back(tex_empty);
        }
        else {
          ccl::float2 attr0, attr1, attr2;
          if (desc.element & (ccl::ATTR_ELEMENT_VERTEX | ccl::ATTR_ELEMENT_VERTEX_MOTION)) {
            attr0 = kernel_data_fetch(attributes_float2, desc.offset + tri_vindex.x);
            attr1 = kernel_data_fetch(attributes_float2, desc.offset + tri_vindex.y);
            attr2 = kernel_data_fetch(attributes_float2, desc.offset + tri_vindex.z);
          }
          else {
            const int tri = desc.offset + prim * 3;
            attr0 = kernel_data_fetch(attributes_float2, tri + 0);
            attr1 = kernel_data_fetch(attributes_float2, tri + 1);
            attr2 = kernel_data_fetch(attributes_float2, tri + 2);
          }

          mini::common::vec2f tex0;
          tex0[0] = attr0[0];
          tex0[1] = attr0[1];
          mini_mesh->texcoords.push_back(tex0);

          mini::common::vec2f tex1;
          tex1[0] = attr1[0];
          tex1[1] = attr1[1];
          mini_mesh->texcoords.push_back(tex1);

          mini::common::vec2f tex2;
          tex2[0] = attr2[0];
          tex2[1] = attr2[1];
          mini_mesh->texcoords.push_back(tex2);
        }
      }
      else {
        mini::common::vec2f tex_empty;
        memset(&tex_empty[0], 0, sizeof(float) * 2);

        mini_mesh->texcoords.push_back(tex_empty);
        mini_mesh->texcoords.push_back(tex_empty);
        mini_mesh->texcoords.push_back(tex_empty);
      }
    }

    if (mini_mesh->indices.size() == 0)
      continue;

    mini::Object::SP mini_object = std::make_shared<mini::Object>();
    mini_object->meshes.push_back(mini_mesh);

    for (ccl::uint obj = temp_scenes[i].start_obj; obj <= temp_scenes[i].end_obj; obj++) {
      ccl::KernelObject kobject = kernel_data_fetch(objects, obj);
      ccl::uint object_flag = kernel_data_fetch(object_flag, obj);

      mini::Instance::SP mini_instance = std::make_shared<mini::Instance>(mini_object);
      // memcpy(&instance->xfm.l.vx, &kobject.tfm, sizeof(float) * 12);

      if (!(object_flag & ccl::SD_OBJECT_TRANSFORM_APPLIED)) {
#  if 0
        mini_instance->xfm.l.vx.x = kobject.tfm.x.x;
        mini_instance->xfm.l.vx.y = kobject.tfm.x.y;
        mini_instance->xfm.l.vx.z = kobject.tfm.x.z;

        mini_instance->xfm.l.vy.x = kobject.tfm.y.x;
        mini_instance->xfm.l.vy.y = kobject.tfm.y.y;
        mini_instance->xfm.l.vy.z = kobject.tfm.y.z;

        mini_instance->xfm.l.vz.x = kobject.tfm.z.x;
        mini_instance->xfm.l.vz.y = kobject.tfm.z.y;
        mini_instance->xfm.l.vz.z = kobject.tfm.z.z;
#  else
        mini_instance->xfm.l.vx.x = kobject.tfm.x.x;
        mini_instance->xfm.l.vy.x = kobject.tfm.x.y;
        mini_instance->xfm.l.vz.x = kobject.tfm.x.z;

        mini_instance->xfm.l.vx.y = kobject.tfm.y.x;
        mini_instance->xfm.l.vy.y = kobject.tfm.y.y;
        mini_instance->xfm.l.vz.y = kobject.tfm.y.z;

        mini_instance->xfm.l.vx.z = kobject.tfm.z.x;
        mini_instance->xfm.l.vy.z = kobject.tfm.z.y;
        mini_instance->xfm.l.vz.z = kobject.tfm.z.z;

#  endif
        mini_instance->xfm.p.x = kobject.tfm.x.w;
        mini_instance->xfm.p.y = kobject.tfm.y.w;
        mini_instance->xfm.p.z = kobject.tfm.z.w;
      }

      mini_scene->instances.push_back(mini_instance);
    }
#endif

  // mini_instances.push_back(instance);
  //  }

  //// fill

  // size_t tidx_size = kg->prim_index.width;

  // for (size_t i = 0; i < tidx_size; i++) {
  //
  //  // check index
  //  if (kernel_data_fetch(prim_index, i) == -1) {
  //    continue;
  //  }

  //  // check visibility
  //  if (kernel_data_fetch(prim_visibility, i) == 0) {
  //    continue;
  //  }

  //  // get object
  //  size_t object = kernel_data_fetch(prim_object, i);

  //  // get tri

  //}

  //// clear
  // for (size_t i = 0; i < mini_instances.size(); i++) {
  //  if (mini_instances[i]->object->meshes[0]->indices.size() > 0)
  //      mini_scene->instances.push_back(mini_instances[i]);
  //}

  // save to file
  mini_scene->save(filename);

  printf("====Statistics: obj: %lld/%lld+%lld, tris: %lld/%lld, verts: %lld/%lld\n",
         obj_size,
         mini_scene->instances.size(),
         mini_obj_size,
         vindex_size,
         mini_vindex_size,
         verts_size,
         mini_verts_size);
}

/////////////////////////////////////////////////////////////////////

void set_kernel_globals(char *kg)
{
  ccl::KernelGlobalsCPU *kernel_globals = (ccl::KernelGlobalsCPU *)kg;

  const char *name = getenv("CLIENT_FILE_KERNEL_GLOBAL");

  if (name == NULL) {
    printf("missing CLIENT_FILE_KERNEL_GLOBAL file\n");
    // return false;
    exit(-1);
  }
  else {
    printf("CLIENT_FILE_KERNEL_GLOBAL file: %s\n", name);
  }

  kg_to_mini((ccl::KernelGlobalsCPU *)kg, name);
}

// CCL_NAMESPACE_END
}  // namespace miniscene
}  // namespace kernel
}  // namespace cyclesphi
