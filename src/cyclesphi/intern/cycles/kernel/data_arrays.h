/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifndef KERNEL_DATA_ARRAY
#  define KERNEL_DATA_ARRAY(type, name)
#endif

/* BVH2, not used for OptiX or Embree. */
KERNEL_DATA_ARRAY(float4, bvh_nodes)
KERNEL_DATA_ARRAY(float4, bvh_leaf_nodes)
KERNEL_DATA_ARRAY(uint, prim_type)
KERNEL_DATA_ARRAY(uint, prim_visibility)
KERNEL_DATA_ARRAY(uint, prim_index)
KERNEL_DATA_ARRAY(uint, prim_object)
KERNEL_DATA_ARRAY(uint, object_node)
KERNEL_DATA_ARRAY(float2, prim_time)

/* objects */
KERNEL_DATA_ARRAY(KernelObject, objects)
KERNEL_DATA_ARRAY(Transform, object_motion_pass)
KERNEL_DATA_ARRAY(DecomposedTransform, object_motion)
KERNEL_DATA_ARRAY(uint, object_flag)
KERNEL_DATA_ARRAY(float, object_volume_step)
KERNEL_DATA_ARRAY(uint, object_prim_offset)
KERNEL_DATA_ARRAY(uint, object_prim_count)

/* cameras */
KERNEL_DATA_ARRAY(DecomposedTransform, camera_motion)

/* triangles */
KERNEL_DATA_ARRAY(uint, tri_shader)
KERNEL_DATA_ARRAY(packed_float3, tri_vnormal)
KERNEL_DATA_ARRAY(uint4, tri_vindex)
KERNEL_DATA_ARRAY(uint, tri_patch)
KERNEL_DATA_ARRAY(float2, tri_patch_uv)
KERNEL_DATA_ARRAY(packed_float3, tri_verts)

/* curves */
KERNEL_DATA_ARRAY(KernelCurve, curves)
KERNEL_DATA_ARRAY(float4, curve_keys)
KERNEL_DATA_ARRAY(KernelCurveSegment, curve_segments)

/* patches */
KERNEL_DATA_ARRAY(uint, patches)

/* pointclouds */
KERNEL_DATA_ARRAY(float4, points)
KERNEL_DATA_ARRAY(uint, points_shader)

/* attributes */
KERNEL_DATA_ARRAY(AttributeMap, attributes_map)
KERNEL_DATA_ARRAY(float, attributes_float)
KERNEL_DATA_ARRAY(float2, attributes_float2)
KERNEL_DATA_ARRAY(packed_float3, attributes_float3)
KERNEL_DATA_ARRAY(float4, attributes_float4)
KERNEL_DATA_ARRAY(uchar4, attributes_uchar4)

/* lights */
KERNEL_DATA_ARRAY(KernelLightDistribution, light_distribution)
KERNEL_DATA_ARRAY(KernelLight, lights)
KERNEL_DATA_ARRAY(float2, light_background_marginal_cdf)
KERNEL_DATA_ARRAY(float2, light_background_conditional_cdf)

/* particles */
KERNEL_DATA_ARRAY(KernelParticle, particles)

/* shaders */
KERNEL_DATA_ARRAY(uint4, svm_nodes)
KERNEL_DATA_ARRAY(KernelShader, shaders)

/* lookup tables */
KERNEL_DATA_ARRAY(float, lookup_table)

/* sobol */
KERNEL_DATA_ARRAY(float, sample_pattern_lut)

/* image textures */
KERNEL_DATA_ARRAY(TextureInfo, texture_info)

/* ies lights */
KERNEL_DATA_ARRAY(float, ies)

KERNEL_DATA_ARRAY(uint, hits)
KERNEL_DATA_ARRAY(uint, bounces)
KERNEL_DATA_ARRAY(uint, diffuse_bounces)
KERNEL_DATA_ARRAY(uint, glossy_bounces)
KERNEL_DATA_ARRAY(uint, transmission_bounces)
KERNEL_DATA_ARRAY(uint, transparent_bounces)
KERNEL_DATA_ARRAY(uint, volume_bounces)
KERNEL_DATA_ARRAY(uint, volume_bounds_bounces)
KERNEL_DATA_ARRAY(uint, bvh_nodes_max_level)

#undef KERNEL_DATA_ARRAY
