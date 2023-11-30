/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#include "device/cpu/kernel_thread_globals.h"

// clang-format off
#include "kernel/osl/shader.h"
#include "kernel/osl/globals.h"
// clang-format on

#ifndef BLENDER_CLIENT
#include "util/profiling.h"
#endif

CCL_NAMESPACE_BEGIN

CPUKernelThreadGlobals::CPUKernelThreadGlobals(const KernelGlobalsCPU &kernel_globals,
                                               void *osl_globals_memory
#ifndef BLENDER_CLIENT
                                               , Profiler &cpu_profiler
#endif
)
    : KernelGlobalsCPU(kernel_globals)
#ifndef BLENDER_CLIENT
    , cpu_profiler_(cpu_profiler)
#endif
{
  reset_runtime_memory();

#ifdef WITH_OSL
  if (osl_globals_memory != NULL)
    OSLShader::thread_init(this, reinterpret_cast<OSLGlobals *>(osl_globals_memory));
#else
  (void)osl_globals_memory;
#endif
}

CPUKernelThreadGlobals::CPUKernelThreadGlobals(CPUKernelThreadGlobals &&other) noexcept
    : KernelGlobalsCPU(std::move(other))
#ifndef BLENDER_CLIENT
    , cpu_profiler_(other.cpu_profiler_)
#endif
{
  other.reset_runtime_memory();
}

CPUKernelThreadGlobals::~CPUKernelThreadGlobals()
{
#ifdef WITH_OSL
  OSLShader::thread_free(this);
#endif
}

CPUKernelThreadGlobals &CPUKernelThreadGlobals::operator=(CPUKernelThreadGlobals &&other)
{
  if (this == &other) {
    return *this;
  }

  *static_cast<KernelGlobalsCPU *>(this) = *static_cast<KernelGlobalsCPU *>(&other);

  other.reset_runtime_memory();

  return *this;
}

void CPUKernelThreadGlobals::reset_runtime_memory()
{
#ifdef WITH_OSL
  osl = nullptr;
#endif
}

#ifndef BLENDER_CLIENT
void CPUKernelThreadGlobals::start_profiling()
{
  cpu_profiler_.add_state(&profiler);
}

void CPUKernelThreadGlobals::stop_profiling()
{
  cpu_profiler_.remove_state(&profiler);
}
#endif

CCL_NAMESPACE_END
