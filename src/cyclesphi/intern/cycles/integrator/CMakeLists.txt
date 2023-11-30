# SPDX-License-Identifier: Apache-2.0
# Copyright 2011-2022 Blender Foundation

set(INC
  ..
  ../../../client/api
  ../../../client/renderengine/src  
)

set(SRC
  adaptive_sampling.cpp
  denoiser.cpp
  denoiser_device.cpp
  denoiser_oidn.cpp
  denoiser_optix.cpp
  path_trace.cpp
  tile.cpp
  pass_accessor.cpp
  pass_accessor_cpu.cpp
  pass_accessor_gpu.cpp
  path_trace_display.cpp
  path_trace_tile.cpp
  path_trace_work.cpp
  path_trace_work_cpu.cpp
  path_trace_work_gpu.cpp
  path_trace_work_client.cpp  
  render_scheduler.cpp
  shader_eval.cpp
  work_balancer.cpp
  work_tile_scheduler.cpp
)

set(SRC_HEADERS
  adaptive_sampling.h
  denoiser.h
  denoiser_device.h
  denoiser_oidn.h
  denoiser_optix.h
  path_trace.h
  tile.h
  pass_accessor.h
  pass_accessor_cpu.h
  pass_accessor_gpu.h
  path_trace_display.h
  path_trace_tile.h
  path_trace_work.h
  path_trace_work_cpu.h
  path_trace_work_gpu.h
  path_trace_work_client.h  
  render_scheduler.h
  shader_eval.h
  work_balancer.h
  work_tile_scheduler.h
)

set(LIB
  cycles_device

  # NOTE: Is required for RenderBuffers access. Might consider moving files around a bit to
  # avoid such cyclic dependency.
  cycles_session

  cycles_util
)

# CLIENT module
if(WITH_CLIENT_MPI)
  add_definitions(-DWITH_CLIENT_MPI)
endif()

if(WITH_CLIENT_MPI_VRCLIENT)
  add_definitions(-DWITH_CLIENT_MPI_VRCLIENT)
endif()

if(WITH_CLIENT_MPI_SOCKET)
  add_definitions(-DWITH_CLIENT_MPI_SOCKET)
endif()

if(WITH_CLIENT_SOCKET)
  add_definitions(-DWITH_CLIENT_SOCKET)
endif()

if(WITH_CLIENT_FILE)
  add_definitions(-DWITH_CLIENT_FILE)
endif()

if(WITH_CLIENT_FILE_MINISCENE)
  add_definitions(-DWITH_CLIENT_FILE_MINISCENE)
endif()

if(WITH_CLIENT_CACHE)
	add_definitions(-DWITH_CLIENT_CACHE)
endif()

if(WITH_CLIENT_DENOISER)
  add_definitions(-DWITH_CLIENT_DENOISER)
endif()

if(WITH_CLIENT_GPUJPEG)
  add_definitions(-DWITH_CLIENT_GPUJPEG)
endif()

if(WITH_CYCLES_DEVICE_CLIENT)
  add_definitions(-DWITH_CLIENT)
endif()

if(WITH_OPENMP)
  add_definitions(-DWITH_OPENMP)
endif()

if(CLIENT_MPI_LOAD_BALANCING_SAMPLES)
  add_definitions(-DCLIENT_MPI_LOAD_BALANCING_SAMPLES)
endif()

if(WITH_CLIENT_RENDERENGINE_SENDER)
  add_definitions(-DWITH_CLIENT_RENDERENGINE_SENDER)
endif()

if(WITH_OPENIMAGEDENOISE)
  list(APPEND LIB
    ${OPENIMAGEDENOISE_LIBRARIES}
  )
endif()

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_integrator "${LIB}" ${SRC} ${SRC_HEADERS})
