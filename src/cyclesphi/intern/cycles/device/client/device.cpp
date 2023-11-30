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
 */

#include "device/client/device.h"
#include "device/client/device_impl.h"


#   include "device/cuda/device.h"
#ifdef WITH_CLIENT_OPTIX
#   include "device/optix/device_impl.h"
//#   include <optix_function_table_definition.h>
#endif

CCL_NAMESPACE_BEGIN

bool device_client_init()
{
#ifdef WITH_CLIENT_OPTIX
  if (g_optixFunctionTable.optixDeviceContextCreate != NULL) {
    /* Already initialized function table. */
    return true;
  }
#endif
  /* Need to initialize CUDA as well. */
  //if (!device_cuda_init()) {
  //  return false;
  //}
  device_cuda_init();
  
#ifdef WITH_CLIENT_OPTIX
  const OptixResult result = optixInit();

  if (result == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION) {
    VLOG(1) << "OptiX initialization failed because the installed NVIDIA driver is too old. "
               "Please update to the latest driver first!";
    return false;
  }
  else if (result != OPTIX_SUCCESS) {
    VLOG(1) << "OptiX initialization failed with error code " << (unsigned int)result;
    return false;
  }

#endif

  /* Loaded Client successfully! */
  return true;
}

Device *device_client_create(const DeviceInfo &info, Stats &stats, Profiler &profiler)
{
  return new CLIENTDevice(info, stats, profiler);
}

void device_client_info(vector<DeviceInfo> &devices)
{
  DeviceInfo info;

  info.type = DEVICE_CLIENT;
  info.description = string_printf("CLIENT");
  info.id = string_printf("CLIENT");
  info.num = 0;
  info.has_osl = false;
  info.has_nanovdb = true;
  info.has_profiling = false;

  devices.insert(devices.begin(), info);
}

string device_client_capabilities()
{
  string capabilities = "";
  return capabilities;
}

CCL_NAMESPACE_END
