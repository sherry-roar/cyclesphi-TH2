/*
 * Copyright 2011-2021 Blender Foundation
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

#pragma once

//#ifdef WITH_CUDA

#  include "device/kernel.h"
#  include "device/memory.h"
#  include "device/queue.h"

//#  include "device/cuda/util.h"

CCL_NAMESPACE_BEGIN

class CLIENTDevice;
class device_memory;

/* Base class for CLIENT queues. */
class CLIENTDeviceQueue : public DeviceQueue {
 public:
  CLIENTDeviceQueue(CLIENTDevice *device);
  ~CLIENTDeviceQueue();

  virtual int num_concurrent_states(const size_t state_size) const override;

  virtual void init_execution() override;

  //virtual bool kernel_available(DeviceKernel kernel) const override;

  virtual bool enqueue(DeviceKernel kernel,
                       const int work_size,
                       DeviceKernelArguments const &args) override;

  virtual bool synchronize() override;

  virtual void zero_to_device(device_memory &mem) override;
  virtual void copy_to_device(device_memory &mem) override;
  virtual void copy_from_device(device_memory &mem) override;

  //virtual CUstream stream()
  //{
  //  return cuda_stream_;
  //}

 protected:
  CLIENTDevice *client_device_;
  //CUstream cuda_stream_;
};

CCL_NAMESPACE_END

//#endif /* WITH_CUDA */
