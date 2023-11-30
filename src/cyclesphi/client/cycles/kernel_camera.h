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
#ifndef __KERNEL_CAMERA_H__
#  define __KERNEL_CAMERA_H__

#  include "cyclesphi_data.h"

namespace cyclesphi {
namespace kernel {

void view_to_kernel_camera(char *kdata, cyclesphi_data *cd);
void bcam_to_kernel_camera(char *kdata, char *bc, int width, int height);
void bcam_to_kernel_camera_right(char *kdata, char *bc, int width, int height);

#if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_CLIENT_ULTRAGRID)
void view_to_kernel_camera_right(char *kdata, cyclesphi_data *cd);
#endif
}
}

#  endif /* __KERNEL_CAMERA_H__ */
