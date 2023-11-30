set(INC
	.
	../../intern/cycles
	#../../intern/cycles/kernel/device/client
	#../../intern/cycles/kernel/device/cpu
	../../intern/atomic
	#../../intern/cycles/render

  ../../source/blender/blenlib
  ../../source/blender/makesdna

  ../../extern/hipew/include
    
	../api
	../renderengine/src
	../ultragrid
    
	${PGI_INCLUDE_DIR}
	${MPI_CXX_HEADER_DIR}
	${CUDA_TOOLKIT_INCLUDE}
	${TURBOJPEG_INCLUDE_DIR}
	${X264_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${NVPIPE_INCLUDE_DIR}
	${ZEROMQ_INCLUDE_DIR}
	${OPTIX_INCLUDE_DIR}
	${ULTRAGRID_INCLUDE_DIR}
  ${GPUJPEG_INCLUDE_DIR}
  ${NCCL_INCLUDE_DIR}
)

set(SRC
	cycles_client.cpp
	cycles_client.h

	kernel_camera.h
	#kernel_camera.cpp
  ../../intern/cycles/blender/camera.cpp

	kernel_util.h
	kernel_util.cpp

	../renderengine/src/cyclesphi_data.h
	../api/client_api.h
)

set(SRC_UTIL
	../../intern/cycles/util/guarded_allocator.cpp
	../../intern/cycles/util/transform.cpp
  ../../intern/cycles/util/string.cpp
  ../../intern/cycles/util/aligned_malloc.cpp
)

set(SRC_KERNEL
	../../intern/cycles/kernel/types.h
)

set(SRC_KERNEL_CLIENT
	#../../intern/cycles/kernel/device/client/kernel_client.cpp
	#../../intern/cycles/kernel/device/client/kernel_client.h
)

if(WITH_CLIENT_CPU)
        add_definitions(-DWITH_CLIENT_CPU)
        list(APPEND SRC
           kernel_omp.h
           kernel_omp.cpp
        )

  set(SRC_CPU_INTEGRATOR
    #kernel_path_trace_work_gpu.h
    ../../intern/cycles/integrator/path_trace_work_cpu.h
    #kernel_path_trace_work_gpu.cpp
    ../../intern/cycles/integrator/path_trace_work_cpu.cpp
    
    ../../intern/cycles/integrator/path_trace_work.h
    ../../intern/cycles/integrator/path_trace_work.cpp

    #kernel_work_tile_scheduler.h
    ../../intern/cycles/integrator/work_tile_scheduler.h
    #kernel_work_tile_scheduler.cpp
    ../../intern/cycles/integrator/work_tile_scheduler.cpp

    ../../intern/cycles/integrator/tile.h
    ../../intern/cycles/integrator/tile.cpp

    #kernel_pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor_cpu.h
    ../../intern/cycles/integrator/pass_accessor.h
    #kernel_pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor_cpu.cpp
    ../../intern/cycles/integrator/pass_accessor.cpp
  )

  set(SRC_CPU_SCENE
    ../../intern/cycles/scene/pass.h
    ../../intern/cycles/scene/pass.cpp

    ../../intern/cycles/scene/scene.h
    ../../intern/cycles/scene/film.h
  )

  set(SRC_CPU_SESSION
    ../../intern/cycles/session/buffers.h
    ../../intern/cycles/session/buffers.cpp
  )

  set(SRC_CPU_DEVICE_CPU
    #kernel_queue.h
    #../../intern/cycles/device/cpu/queue.h
    #kernel_queue.cpp
    #../../intern/cycles/device/cpu/queue.cpp
    
    ../../intern/cycles/device/cpu/kernel.h
    ../../intern/cycles/device/cpu/kernel.cpp
    
    ../../intern/cycles/device/cpu/kernel_thread_globals.h
    ../../intern/cycles/device/cpu/kernel_thread_globals.cpp
    
    ../../intern/cycles/device/cpu/device_impl.h
    ../../intern/cycles/device/cpu/device_impl.cpp              
  )

  set(SRC_CPU_DEVICE    
    ../../intern/cycles/device/kernel.h
    ../../intern/cycles/device/kernel.cpp
    
    ../../intern/cycles/device/queue.h
    ../../intern/cycles/device/queue.cpp
    
    ../../intern/cycles/device/device.h
    ../../intern/cycles/device/device.cpp
    
    ../../intern/cycles/device/memory.h
    ../../intern/cycles/device/memory.cpp
  )

  set(SRC_CPU_KERNEL_CPU   
    ../../intern/cycles/kernel/device/cpu/kernel.h
    ../../intern/cycles/kernel/device/cpu/kernel.cpp

    ../../intern/cycles/kernel/device/cpu/kernel_avx.cpp
    ../../intern/cycles/kernel/device/cpu/kernel_avx2.cpp
    ../../intern/cycles/kernel/device/cpu/kernel_sse2.cpp
    ../../intern/cycles/kernel/device/cpu/kernel_sse3.cpp
    ../../intern/cycles/kernel/device/cpu/kernel_sse41.cpp
  )

  list(APPEND SRC
    ${SRC_CPU_CLIENT}
    ${SRC_CPU_INTEGRATOR}
    ${SRC_CPU_SCENE}
    ${SRC_CPU_SESSION}
    ${SRC_CPU_DEVICE}
    ${SRC_CPU_DEVICE_CPU}
    ${SRC_CPU_KERNEL_CPU}
  )

  #source_group("cycles\\client" FILES ${SRC_CUDA_CLIENT})
  source_group("blender\\intern\\cycles\\integrator" FILES ${SRC_CPU_INTEGRATOR})
  source_group("blender\\intern\\cycles\\scene" FILES ${SRC_CPU_SCENE})
  source_group("blender\\intern\\cycles\\session" FILES ${SRC_CPU_SESSION})
  source_group("blender\\intern\\cycles\\device" FILES ${SRC_CPU_DEVICE})
  source_group("blender\\intern\\cycles\\device\\cpu" FILES ${SRC_CPU_DEVICE_CPU})
  source_group("blender\\intern\\cycles\\kernel\\device\\cpu" FILES ${SRC_CPU_KERNEL_CPU})
endif()

if(WITH_CLIENT_CUDA)
  add_definitions(-DWITH_CLIENT_CUDA)
  add_definitions(-DWITH_CUDA)
  set(SRC_CUDA_CLIENT
    kernel_cuda.h
    kernel_cuda_stream.cpp

    #kernel_cuda_device.h
    #kernel_cuda_device.cpp
    
    kernel_cuda_util.h
    kernel_cuda_util.cpp

    kernel_cuda_stat.h
    kernel_cuda_stat.cpp

    #kernel_cuda_context_scope.h
    #kernel_cuda_context_scope.cpp
  )

  set(SRC_CUDA_INTEGRATOR
    #kernel_path_trace_work_gpu.h
    ../../intern/cycles/integrator/path_trace_work_gpu.h
    #kernel_path_trace_work_gpu.cpp
    ../../intern/cycles/integrator/path_trace_work_gpu.cpp
    
    ../../intern/cycles/integrator/path_trace_work.h
    ../../intern/cycles/integrator/path_trace_work.cpp

    #kernel_work_tile_scheduler.h
    ../../intern/cycles/integrator/work_tile_scheduler.h
    #kernel_work_tile_scheduler.cpp
    ../../intern/cycles/integrator/work_tile_scheduler.cpp

    ../../intern/cycles/integrator/tile.h
    ../../intern/cycles/integrator/tile.cpp

    #kernel_pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor.h
    #kernel_pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor.cpp
  )

  set(SRC_CUDA_SCENE
    ../../intern/cycles/scene/pass.h
    ../../intern/cycles/scene/pass.cpp

    ../../intern/cycles/scene/scene.h
    ../../intern/cycles/scene/film.h
  )

  set(SRC_CUDA_SESSION
    ../../intern/cycles/session/buffers.h
    ../../intern/cycles/session/buffers.cpp
  )

  set(SRC_CUDA_DEVICE_CUDA
    #kernel_queue.h
    ../../intern/cycles/device/cuda/queue.h
    #kernel_queue.cpp
    ../../intern/cycles/device/cuda/queue.cpp
    
    ../../intern/cycles/device/cuda/kernel.h
    ../../intern/cycles/device/cuda/kernel.cpp
    
    ../../intern/cycles/device/cuda/util.h
    ../../intern/cycles/device/cuda/util.cpp
    
    ../../intern/cycles/device/cuda/device_impl.h
    ../../intern/cycles/device/cuda/device_impl.cpp              
  )

  set(SRC_CUDA_DEVICE    
    ../../intern/cycles/device/kernel.h
    ../../intern/cycles/device/kernel.cpp
    
    ../../intern/cycles/device/queue.h
    ../../intern/cycles/device/queue.cpp
    
    ../../intern/cycles/device/device.h
    ../../intern/cycles/device/device.cpp
    
    ../../intern/cycles/device/memory.h
    ../../intern/cycles/device/memory.cpp

    ../../intern/cycles/device/denoise.h
    ../../intern/cycles/device/denoise.cpp
  )

  list(APPEND SRC
    ${SRC_CUDA_CLIENT}
    ${SRC_CUDA_INTEGRATOR}
    ${SRC_CUDA_SCENE}
    ${SRC_CUDA_SESSION}
    ${SRC_CUDA_DEVICE}
    ${SRC_CUDA_DEVICE_CUDA}
  )

  #source_group("cycles\\client" FILES ${SRC_CUDA_CLIENT})
  source_group("blender\\intern\\cycles\\integrator" FILES ${SRC_CUDA_INTEGRATOR})
  source_group("blender\\intern\\cycles\\scene" FILES ${SRC_CUDA_SCENE})
  source_group("blender\\intern\\cycles\\session" FILES ${SRC_CUDA_SESSION})
  source_group("blender\\intern\\cycles\\device" FILES ${SRC_CUDA_DEVICE})
  source_group("blender\\intern\\cycles\\device\\cuda" FILES ${SRC_CUDA_DEVICE_CUDA})
  
endif()

if(WITH_CLIENT_OPTIX)
  add_definitions(-DWITH_CLIENT_OPTIX)
  add_definitions(-DWITH_OPTIX)
  add_definitions(-DWITH_CUDA)
  set(SRC_OPTIX_CLIENT
    kernel_optix.h
    kernel_optix.cpp

    kernel_cuda.h
    kernel_cuda_stream.cpp

    kernel_cuda_util.h
    kernel_cuda_util.cpp

    kernel_cuda_stat.h
    kernel_cuda_stat.cpp
  )

  set(SRC_OPTIX_INTEGRATOR
    #kernel_path_trace_work_gpu.h
    ../../intern/cycles/integrator/path_trace_work_gpu.h
    #kernel_path_trace_work_gpu.cpp
    ../../intern/cycles/integrator/path_trace_work_gpu.cpp
    
    ../../intern/cycles/integrator/path_trace_work.h
    ../../intern/cycles/integrator/path_trace_work.cpp

    #kernel_work_tile_scheduler.h
    ../../intern/cycles/integrator/work_tile_scheduler.h
    #kernel_work_tile_scheduler.cpp
    ../../intern/cycles/integrator/work_tile_scheduler.cpp

    ../../intern/cycles/integrator/tile.h
    ../../intern/cycles/integrator/tile.cpp

    #kernel_pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor.h
    #kernel_pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor.cpp
  )

  set(SRC_OPTIX_SCENE
    ../../intern/cycles/scene/pass.h
    ../../intern/cycles/scene/pass.cpp

    ../../intern/cycles/scene/scene.h
    ../../intern/cycles/scene/film.h
  )

  set(SRC_OPTIX_SESSION
    ../../intern/cycles/session/buffers.h
    ../../intern/cycles/session/buffers.cpp
  )

  set(SRC_OPTIX_DEVICE_CUDA
    #kernel_queue.h
    ../../intern/cycles/device/cuda/queue.h
    #kernel_queue.cpp
    ../../intern/cycles/device/cuda/queue.cpp
    
    ../../intern/cycles/device/cuda/kernel.h
    ../../intern/cycles/device/cuda/kernel.cpp
    
    ../../intern/cycles/device/cuda/util.h
    ../../intern/cycles/device/cuda/util.cpp
    
    ../../intern/cycles/device/cuda/device_impl.h
    ../../intern/cycles/device/cuda/device_impl.cpp              
  )

  set(SRC_OPTIX_DEVICE_OPTIX
    #kernel_queue.h
    ../../intern/cycles/device/optix/queue.h
    #kernel_queue.cpp
    ../../intern/cycles/device/optix/queue.cpp
    
    #../../intern/cycles/device/optix/kernel.h
    #../../intern/cycles/device/optix/kernel.cpp
    
    #../../intern/cycles/device/optix/util.h
    #../../intern/cycles/device/optix/util.cpp
    
    ../../intern/cycles/device/optix/device_impl.h
    ../../intern/cycles/device/optix/device_impl.cpp              
  )

  set(SRC_OPTIX_DEVICE    
    ../../intern/cycles/device/kernel.h
    ../../intern/cycles/device/kernel.cpp
    
    ../../intern/cycles/device/queue.h
    ../../intern/cycles/device/queue.cpp
    
    ../../intern/cycles/device/device.h
    ../../intern/cycles/device/device.cpp
    
    ../../intern/cycles/device/memory.h
    ../../intern/cycles/device/memory.cpp

    ../../intern/cycles/device/denoise.h
    ../../intern/cycles/device/denoise.cpp
  )

  list(APPEND SRC
    ${SRC_OPTIX_CLIENT}
    ${SRC_OPTIX_INTEGRATOR}
    ${SRC_OPTIX_SCENE}
    ${SRC_OPTIX_SESSION}
    ${SRC_OPTIX_DEVICE}
    ${SRC_OPTIX_DEVICE_OPTIX}
    ${SRC_OPTIX_DEVICE_CUDA}
  )

  #source_group("cycles\\client" FILES ${SRC_OPTIX_CLIENT})
  source_group("blender\\intern\\cycles\\integrator" FILES ${SRC_OPTIX_INTEGRATOR})
  source_group("blender\\intern\\cycles\\scene" FILES ${SRC_OPTIX_SCENE})
  source_group("blender\\intern\\cycles\\session" FILES ${SRC_OPTIX_SESSION})
  source_group("blender\\intern\\cycles\\device" FILES ${SRC_OPTIX_DEVICE})
  source_group("blender\\intern\\cycles\\device\\optix" FILES ${SRC_OPTIX_DEVICE_OPTIX})
  source_group("blender\\intern\\cycles\\device\\cuda" FILES ${SRC_OPTIX_DEVICE_CUDA})
  
endif()

if(WITH_CLIENT_HIP)
  add_definitions(-DWITH_CLIENT_HIP)
  add_definitions(-DWITH_HIP)
  add_definitions(-DWITH_HIP_DYNLOAD)
  set(SRC_HIP_CLIENT
    kernel_hip.h
	  kernel_hip.cpp

    #kernel_path_trace_work_gpu.h
    #kernel_path_trace_work_gpu.cpp

    #kernel_queue.h
    #kernel_queue.cpp

    #kernel_work_tile_scheduler.h
    #kernel_work_tile_scheduler.cpp

    #kernel_hip_device.h
    #kernel_hip_device.cpp

    kernel_hip_util.h
    kernel_hip_util.cpp

    kernel_hip_stat.h
    kernel_hip_stat.cpp

    #kernel_hip_context_scope.h
    #kernel_hip_context_scope.cpp

    #kernel_pass_accessor_gpu.cpp
    #kernel_pass_accessor_gpu.h
  )

  set(SRC_HIP_INTEGRATOR
    #kernel_path_trace_work_gpu.h
    ../../intern/cycles/integrator/path_trace_work_gpu.h
    #kernel_path_trace_work_gpu.cpp
    ../../intern/cycles/integrator/path_trace_work_gpu.cpp
    
    ../../intern/cycles/integrator/path_trace_work.h
    ../../intern/cycles/integrator/path_trace_work.cpp

    #kernel_work_tile_scheduler.h
    ../../intern/cycles/integrator/work_tile_scheduler.h
    #kernel_work_tile_scheduler.cpp
    ../../intern/cycles/integrator/work_tile_scheduler.cpp

    ../../intern/cycles/integrator/tile.h
    ../../intern/cycles/integrator/tile.cpp

    #kernel_pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor_gpu.h
    ../../intern/cycles/integrator/pass_accessor.h
    #kernel_pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor_gpu.cpp
    ../../intern/cycles/integrator/pass_accessor.cpp
  )

  set(SRC_HIP_SCENE
    ../../intern/cycles/scene/pass.h
    ../../intern/cycles/scene/pass.cpp

    ../../intern/cycles/scene/scene.h
    ../../intern/cycles/scene/film.h
  )

  set(SRC_HIP_SESSION
    ../../intern/cycles/session/buffers.h
    ../../intern/cycles/session/buffers.cpp
  )

  set(SRC_HIP_DEVICE_HIP
    #kernel_queue.h
    ../../intern/cycles/device/hip/queue.h
    #kernel_queue.cpp
    ../../intern/cycles/device/hip/queue.cpp
    
    ../../intern/cycles/device/hip/kernel.h
    ../../intern/cycles/device/hip/kernel.cpp
    
    ../../intern/cycles/device/hip/util.h
    ../../intern/cycles/device/hip/util.cpp
    
    ../../intern/cycles/device/hip/device_impl.h
    ../../intern/cycles/device/hip/device_impl.cpp              
  )

  set(SRC_HIP_DEVICE    
    ../../intern/cycles/device/kernel.h
    ../../intern/cycles/device/kernel.cpp
    
    ../../intern/cycles/device/queue.h
    ../../intern/cycles/device/queue.cpp
    
    ../../intern/cycles/device/device.h
    ../../intern/cycles/device/device.cpp
    
    ../../intern/cycles/device/memory.h
    ../../intern/cycles/device/memory.cpp
  )

  list(APPEND SRC
    ${SRC_HIP_CLIENT}
    ${SRC_HIP_INTEGRATOR}
    ${SRC_HIP_SCENE}
    ${SRC_HIP_SESSION}
    ${SRC_HIP_DEVICE}
    ${SRC_HIP_DEVICE_HIP}
  )

  #source_group("cycles\\client" FILES ${SRC_HIP_CLIENT})
  source_group("blender\\intern\\cycles\\integrator" FILES ${SRC_HIP_INTEGRATOR})
  source_group("blender\\intern\\cycles\\scene" FILES ${SRC_HIP_SCENE})
  source_group("blender\\intern\\cycles\\session" FILES ${SRC_HIP_SESSION})
  source_group("blender\\intern\\cycles\\device" FILES ${SRC_HIP_DEVICE})
  source_group("blender\\intern\\cycles\\device\\hip" FILES ${SRC_HIP_DEVICE_HIP})

endif()

add_definitions(
	-DCCL_NAMESPACE_BEGIN=namespace\ ccl\ {
	-DCCL_NAMESPACE_END=}
)

add_definitions(-DBLENDER_CLIENT)

if(WITH_CPU_SIMD)
	add_definitions(
		-DWITH_CPU_SIMD
		-DWITH_KERNEL_SSE2
		-DWITH_KERNEL_SSE3
		-DWITH_KERNEL_SSE41
	)
endif()

if(WITH_CPU_AVX)
        add_definitions(
                -DWITH_CPU_AVX
                -DWITH_KERNEL_AVX
        )
endif()

if(WITH_CPU_AVX2)
        add_definitions(
                -DWITH_CPU_AVX2
                -DWITH_KERNEL_AVX2
        )
endif()

if(WITH_OPENMP)
        add_definitions(-DWITH_OPENMP)
endif()

if(WITH_WEBRTC)
    add_definitions(-DWITH_WEBRTC)
endif()                          

if(WITH_CLIENT_UNIMEM)
        add_definitions(-DWITH_CLIENT_UNIMEM)
endif()

if(WITH_CLIENT_FILE)
    add_definitions(-DWITH_CLIENT_FILE)

	list(APPEND SRC_KERNEL_CLIENT
			../../intern/cycles/kernel/device/client/kernel_file.cpp
      ../../intern/cycles/kernel/device/client/kernel_file.h
			)

endif()

if(WITH_CLIENT_MPI)
		add_definitions(-DWITH_CLIENT_MPI)
		list(APPEND SRC_KERNEL_CLIENT
        ../../intern/cycles/kernel/device/client/kernel_tcp.cpp
        ../../intern/cycles/kernel/device/client/kernel_tcp.h

				../../intern/cycles/kernel/device/client/kernel_mpi.cpp
        ../../intern/cycles/kernel/device/client/kernel_mpi.h
				)
endif()

if(WITH_CLIENT_MPI_FILE)
	add_definitions(-DWITH_CLIENT_MPI_FILE)
endif()

if(WITH_CLIENT_MPI_FILE_1)
	add_definitions(-DWITH_CLIENT_MPI_FILE_1)
endif()

if(WITH_CLIENT_MPI_VRCLIENT)
	add_definitions(-DWITH_CLIENT_MPI_VRCLIENT)
endif()

if(WITH_CLIENT_MPI_SOCKET)
	add_definitions(-DWITH_CLIENT_MPI_SOCKET)
        list(APPEND SRC_KERNEL_CLIENT
                ../../intern/cycles/kernel/device/client/kernel_tcp.cpp
                ../../intern/cycles/kernel/device/client/kernel_tcp.h

                ../../intern/cycles/kernel/device/client/kernel_mpi.cpp
                ../../intern/cycles/kernel/device/client/kernel_mpi.h
        )

endif()

if(WITH_CLIENT_NCCL_SOCKET)
	add_definitions(-DWITH_CLIENT_NCCL_SOCKET)
endif()

if(WITH_CLIENT_MPI_REDUCE)
	add_definitions(-DWITH_CLIENT_MPI_REDUCE)
endif()

if(WITH_CLIENT_SOCKET)
	add_definitions(-DWITH_CLIENT_SOCKET)
        list(APPEND SRC_KERNEL_CLIENT
                ../../intern/cycles/kernel/device/client/kernel_tcp.cpp
                ../../intern/cycles/kernel/device/client/kernel_tcp.h

                ../../intern/cycles/kernel/device/client/kernel_socket.cpp
                ../../intern/cycles/kernel/device/client/kernel_socket.h
                )

endif()

if(WITH_CLIENT_RENDERENGINE)
	add_definitions(-DWITH_CLIENT_RENDERENGINE)
        list(APPEND SRC_KERNEL_CLIENT
                ../../intern/cycles/kernel/device/client/kernel_tcp.cpp
                ../../intern/cycles/kernel/device/client/kernel_tcp.h
                )

endif()

if(WITH_CLIENT_RENDERENGINE_EMULATE)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_EMULATE)
endif()

if(WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD)
endif()

if(WITH_CLIENT_RENDERENGINE_VR)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_VR)
endif()

if(WITH_CLIENT_ULTRAGRID_LIB)
	add_definitions(-DWITH_CLIENT_ULTRAGRID_LIB)
endif()

if(WITH_CLIENT_RENDERENGINE_VRCLIENT)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_VRCLIENT)
endif()

if(WITH_CPU_STAT)
	add_definitions(-DWITH_CPU_STAT)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT)
	add_definitions(-DWITH_CLIENT_CUDA_CPU_STAT)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT2)
	add_definitions(-DWITH_CLIENT_CUDA_CPU_STAT2)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT2v2)
	add_definitions(-DWITH_CLIENT_CUDA_CPU_STAT2v2)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT2_LB)
	add_definitions(-DWITH_CLIENT_CUDA_CPU_STAT2_LB)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT3)
	add_definitions(-DWITH_CLIENT_CUDA_CPU_STAT3)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT_LB)
	add_definitions(-DWITH_CLIENT_CUDA_CPU_STAT_LB)
endif()

#  define UNITEST1_ROUND_ROBIN
if(UNITEST1_ROUND_ROBIN)
	add_definitions(-DUNITEST1_ROUND_ROBIN)
endif()
if(UNITEST1_ROUND_ROBIN2)
	add_definitions(-DUNITEST1_ROUND_ROBIN2)
endif()
#define UNITEST1_RANDOM
if(UNITEST1_RANDOM)
	add_definitions(-DUNITEST1_RANDOM)
endif()
#define UNITEST1_COUNTINOUS
if(UNITEST1_COUNTINOUS)
	add_definitions(-DUNITEST1_COUNTINOUS)
endif()

#  define UNITEST1_CPU
if(UNITEST1_CPU)
	add_definitions(-DUNITEST1_CPU)
endif()
#  define UNITEST1_CREDITS_FILE
if(UNITEST1_CREDITS_FILE)
	add_definitions(-DUNITEST1_CREDITS_FILE)
endif()

#  define UNITEST2_ROUND_ROBIN
if(UNITEST2_ROUND_ROBIN)
	add_definitions(-DUNITEST2_ROUND_ROBIN)
endif()
#define UNITEST2_RANDOM
if(UNITEST2_RANDOM)
	add_definitions(-DUNITEST2_RANDOM)
endif()

#  define UNITEST2_CREDITS
if(UNITEST2_CREDITS)
	add_definitions(-DUNITEST2_CREDITS)
endif()
#  define UNITEST2_CREDITS_CPU
if(UNITEST2_CREDITS_CPU)
	add_definitions(-DUNITEST2_CREDITS_CPU)
endif()

if(UNITEST2_CREDITS_CPU2)
	add_definitions(-DUNITEST2_CREDITS_CPU2)
endif()

if(WITH_IMAGE_TURBOJPEG)
	add_definitions(-DWITH_TURBOJPEG)
endif()

if(WITH_IMAGE_X264)
	add_definitions(-DWITH_X264)
endif()

if(WITH_NVPIPE)
	add_definitions(-DWITH_NVPIPE)
endif()

if(WITH_CLIENT_GPUJPEG)
	add_definitions(-DWITH_CLIENT_GPUJPEG)
endif()

if(WITH_KNC_NATIVE_GCC)
	add_definitions(-D__KERNEL_MIC_GCC__)
endif()

if(WITH_CLIENT_FILE_MMAP)
	add_definitions(-DWITH_CLIENT_FILE_MMAP)
endif()

if(ENABLE_LOAD_BALANCE)
	add_definitions(-DENABLE_LOAD_BALANCE)
endif()

if(ENABLE_LOAD_BALANCEv2)
	add_definitions(-DENABLE_LOAD_BALANCEv2)
endif()

if(ENABLE_LOAD_BALANCEv3)
	add_definitions(-DENABLE_LOAD_BALANCEv3)
endif()

if(ENABLE_LOAD_BALANCE_EXIT)
	add_definitions(-DENABLE_LOAD_BALANCE_EXIT)
endif()

if(ENABLE_LOAD_BALANCE_CUDA)
	add_definitions(-DENABLE_LOAD_BALANCE_CUDA)
endif()

if(ENABLE_INC_SAMPLES)
	add_definitions(-DENABLE_INC_SAMPLES)
endif()

if(WITH_CUDA_CPUIMAGE)
	add_definitions(-DWITH_CUDA_CPUIMAGE)
  		set(SRC_KERNEL_CPU
				../../intern/cycles/kernel/device/cpu/image.h
				)
endif()

if(WITH_HIP_CPUIMAGE)
	add_definitions(-DWITH_HIP_CPUIMAGE)
  		set(SRC_KERNEL_CPU
				../../intern/cycles/kernel/device/cpu/image.h
				)
endif()

if(WITH_CUDA_STAT)
	add_definitions(-DWITH_CUDA_STAT)
endif()

if(WITH_CUDA_STATv2)
	add_definitions(-DWITH_CUDA_STATv2)
endif()

if(WITH_CUDA_STATv2_LB)
  add_definitions(-DWITH_CUDA_STATv2_LB)
endif()

if(WITH_HIP_STAT)
	add_definitions(-DWITH_HIP_STAT)
endif()

if(WITH_HIP_STATv2)
	add_definitions(-DWITH_HIP_STATv2)
endif()

if(WITH_HIP_STATv2_LB)
  add_definitions(-DWITH_HIP_STATv2_LB)
endif()

if(WITH_SOCKET_UDP)
	add_definitions(-DWITH_SOCKET_UDP)
endif()

if(WITH_SOCKET_ONLY_DATA)
	add_definitions(-DWITH_SOCKET_ONLY_DATA)
endif()

if(CLIENT_MPI_LOAD_BALANCING_SAMPLES)
	add_definitions(-DCLIENT_MPI_LOAD_BALANCING_SAMPLES)
endif()

if(CLIENT_MPI_LOAD_BALANCING_LINES)
	add_definitions(-DCLIENT_MPI_LOAD_BALANCING_LINES)
endif()

if(WITH_LOAD_BALANCING_COLUMN)
	add_definitions(-DWITH_LOAD_BALANCING_COLUMN)
endif()

if(WITH_CLIENT_SHOW_STAT)
	add_definitions(-DWITH_CLIENT_SHOW_STAT)
endif()

if(WITH_CLIENT_SHOW_STAT_BVH_LOOP)
	add_definitions(-DWITH_CLIENT_SHOW_STAT_BVH_LOOP)
endif()

if(WITH_RGBA_HALF)
	add_definitions(-DWITH_RGBA_HALF)
endif()

if(WITH_CLIENT_BUFFER_DECODER)
	add_definitions(-DWITH_CLIENT_BUFFER_DECODER)
endif()

if(WITH_CLIENT_CACHE)
	add_definitions(-DWITH_CLIENT_CACHE)
endif()

if(WITH_CLIENT_MPI_CACHE)
	add_definitions(-DWITH_CLIENT_MPI_CACHE)
endif()

if(WITH_OIDN_DENOISER)
	add_definitions(-DWITH_OIDN_DENOISER)
endif()

if(WITH_OPTIX_DENOISER)
  add_definitions(-DWITH_OPTIX_DENOISER)
endif()

if(WITH_POP)
  add_definitions(-DWITH_POP)

  if(CLIENT_PATH_TRACE_ROW)
    add_definitions(-DCLIENT_PATH_TRACE_ROW)
  endif()
  if(CLIENT_PATH_TRACE_COLUMN)
    add_definitions(-DCLIENT_PATH_TRACE_COLUMN)
  endif()
  if(CLIENT_PATH_TRACE_TILE)
   add_definitions(-DCLIENT_PATH_TRACE_TILE)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_1)
   add_definitions(-DCLIENT_PATH_TRACE_PIXEL_1)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_2)
    add_definitions(-DCLIENT_PATH_TRACE_PIXEL_2)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_4)
   add_definitions(-DCLIENT_PATH_TRACE_PIXEL_4)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_8)
   add_definitions(-DCLIENT_PATH_TRACE_PIXEL_8)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_16)
   add_definitions(-DCLIENT_PATH_TRACE_PIXEL_16)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_32)
   add_definitions(-DCLIENT_PATH_TRACE_PIXEL_32)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_64)
    add_definitions(-DCLIENT_PATH_TRACE_PIXEL_64)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_128)
    add_definitions(-DCLIENT_PATH_TRACE_PIXEL_128)
  endif()
  if(CLIENT_PATH_TRACE_PIXEL_256)
    add_definitions(-DCLIENT_PATH_TRACE_PIXEL_256)
  endif()


endif()

if(WITH_CLIENT_ULTRAGRID)
	add_definitions(-DWITH_CLIENT_ULTRAGRID)
endif()

if(WITH_RGBA_FORMAT)
	add_definitions(-DWITH_RGBA_FORMAT)
endif()

if(WITH_CLIENT_XOR_RLE)
	add_definitions(-DWITH_CLIENT_XOR_RLE)
endif()

if(WITH_CLIENT_YUV)
	add_definitions(-DWITH_CLIENT_YUV)
endif()

if(WITH_VRCLIENT_XTAL)
	add_definitions(-DWITH_VRCLIENT_XTAL)
endif()

if(WITH_SIZE_UCHAR3)
	add_definitions(-DWITH_SIZE_UCHAR3)
endif()

if(WITH_CUDA_BUFFER_MANAGED)
	add_definitions(-DWITH_CUDA_BUFFER_MANAGED)
endif()

if(WITH_ORIG_TILES)
	add_definitions(-DWITH_ORIG_TILES)
endif()

if(WITH_VRCLIENT_RECEIVER)
	add_definitions(-DWITH_VRCLIENT_RECEIVER)
endif()

if(ENABLE_STEP_SAMPLES)
	add_definitions(-DENABLE_STEP_SAMPLES)
endif()

if(WITH_CLIENT_CUDA_GPU_TILES)
	add_definitions(-DWITH_CLIENT_CUDA_GPU_TILES)
endif()

if(WITH_CLIENT_FORCE_CUDA)
	add_definitions(-DWITH_CLIENT_FORCE_CUDA)
endif()

list(APPEND SRC
  ${SRC_UTIL}
  ${SRC_KERNEL}
  ${SRC_KERNEL_CLIENT}
  ${SRC_KERNEL_CPU}
)

source_group("blender\\intern\\cycles\\util" FILES ${SRC_UTIL})
source_group("blender\\intern\\cycles\\kernel" FILES ${SRC_KERNEL})
source_group("blender\\intern\\cycles\\kernel\\device\\client" FILES ${SRC_KERNEL_CLIENT})
source_group("blender\\intern\\cycles\\kernel\\device\\cpu" FILES ${SRC_KERNEL_CPU})

include_directories(${INC})
add_library(cycles${CLIENT_SUFFIX} STATIC ${SRC})

#if(WITH_CLIENT_CUDA)
#	add_dependencies(cycles${CLIENT_SUFFIX} cycles_kernel_cuda)
#endif()

if(WITH_CLIENT_ULTRAGRID)
	add_dependencies(cycles${CLIENT_SUFFIX} ultragrid${CLIENT_SUFFIX})
	#target_link_libraries(cycles${CLIENT_SUFFIX} ultragrid${CLIENT_SUFFIX} blenlib${CLIENT_SUFFIX})
endif()

install (TARGETS cycles${CLIENT_SUFFIX} DESTINATION lib)
