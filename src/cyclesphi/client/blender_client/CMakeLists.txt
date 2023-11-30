
set(INC
	.
	../../../intern/cycles
	../../../intern/cycles/kernel/device/client
	../api
  ../renderengine/src/
	../cycles
	${MPI_CXX_HEADER_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${NCCL_INCLUDE_DIR}
)

set(SRC
	blender_client.cpp
)

set(SRC_HEADERS

)

if(WITH_OPENMP)
	add_definitions(-DWITH_OPENMP)
endif()

if(WITH_NVML)
	add_definitions(-DWITH_NVML)
endif()

if(WITH_CLIENT_FILE)
	add_definitions(-DWITH_CLIENT_FILE)
endif()

if(WITH_CLIENT_MPI)
	add_definitions(-DWITH_CLIENT_MPI)
endif()

if(WITH_CLIENT_MPI_SOCKET)
	add_definitions(-DWITH_CLIENT_MPI_SOCKET)
endif()

if(WITH_CLIENT_NCCL_SOCKET)
	add_definitions(-DWITH_CLIENT_NCCL_SOCKET)
endif()

if(WITH_CLIENT_SOCKET)
	add_definitions(-DWITH_CLIENT_SOCKET)
endif()

if(WITH_CLIENT_RENDERENGINE)
	add_definitions(-DWITH_CLIENT_RENDERENGINE)
endif()

if(WITH_CLIENT_RENDERENGINE_VR)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_VR)
endif()

if(WITH_CLIENT_RENDERENGINE_EMULATE)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_EMULATE)
endif()

if(WITH_CLIENT_MPI_FILE)
	add_definitions(-DWITH_CLIENT_MPI_FILE)
endif()

if(WITH_CLIENT_MPI_FILE_1)
	add_definitions(-DWITH_CLIENT_MPI_FILE_1)
endif()

if(WITH_CLIENT_CUDA)
	add_definitions(-DWITH_CLIENT_CUDA)
endif()

if(WITH_CLIENT_OPTIX)
	add_definitions(-DWITH_CLIENT_OPTIX)
endif()

if(WITH_CLIENT_HIP)
	add_definitions(-DWITH_CLIENT_HIP)
endif()

if(WITH_CLIENT_MPI_VRCLIENT)
	add_definitions(-DWITH_CLIENT_MPI_VRCLIENT)
endif()

if(WITH_NVPIPE)
	add_definitions(-DWITH_NVPIPE)
endif()

if(WITH_CLIENT_GPUJPEG)
	add_definitions(-DWITH_CLIENT_GPUJPEG)
endif()

if(WITH_CLIENT_CACHE)
	add_definitions(-DWITH_CLIENT_CACHE)
endif()

if(WITH_CLIENT_MPI_CACHE)
	add_definitions(-DWITH_CLIENT_MPI_CACHE)
endif()

if(WITH_WEBRTC)
    add_definitions(-DWITH_WEBRTC)
endif()

# if(WITH_CLIENT_GPU_PER_MPI_RANK)
# 	add_definitions(-DWITH_CLIENT_CUDA_PER_MPI_RANK)
# endif()

if(WITH_CLIENT_CUDA_GPU_TILES)
    add_definitions(-DWITH_CLIENT_CUDA_GPU_TILES)
endif()

add_definitions(
	-DCCL_NAMESPACE_BEGIN=namespace\ ccl\ {
	-DCCL_NAMESPACE_END=}
)

add_definitions(-DBLENDER_CLIENT)

include_directories(${INC})

add_executable(blender_client${CLIENT_SUFFIX} ${SRC} ${SRC_HEADERS})
target_link_libraries(blender_client${CLIENT_SUFFIX} cycles${CLIENT_SUFFIX} ${TURBOJPEG_LIBRARIES} ${X264_LIBRARIES} ${MPI_msmpi_LIBRARY} ${ZEROMQ_LIBRARIES})
#message(${ZEROMQ_LIBRARIES})

if(WITH_CLIENT_CUDA OR WITH_CLIENT_OPTIX)
	#${CUDA_CUSTOM_LIBRARIES}
	#${CUDA_CUDA_LIBRARY}  ${CUDA_CUDART_LIBRARY}
	if (CUDA_CUSTOM_LIBRARIES STREQUAL "")
		set(CUDA_CUSTOM_LIBRARIES		 
			${CUDA_CUDA_LIBRARY}
			${CUDA_CUDART_LIBRARY}
		)
		#message(${CUDA_CUSTOM_LIBRARIES})
	endif()	
if(WIN32)
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${CUDA_CUSTOM_LIBRARIES})
else()
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${CUDA_CUSTOM_LIBRARIES} rt)
endif()
	

	#add_dependencies(blender_client${CLIENT_SUFFIX} cycles_kernel_cuda)
endif()

if(WITH_CLIENT_HIP)
  if(WIN32)
	  target_link_libraries(blender_client${CLIENT_SUFFIX} hipew version)
  else()
	  target_link_libraries(blender_client${CLIENT_SUFFIX} hipew dl)
  endif()	
endif()

#if(WITH_VRCLIENT)
target_link_libraries(blender_client${CLIENT_SUFFIX} blenlib)
#endif()

if(WITH_CLIENT_MPI_GCC OR WITH_CLIENT_NCCL_SOCKET)
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${MPI_LIBRARIES})
endif()

if(WITH_CLIENT_NCCL_SOCKET)
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${NCCL_LIBRARIES})
endif()

if(WITH_CLIENT_ULTRAGRID)
	target_link_libraries(blender_client${CLIENT_SUFFIX} ultragrid${CLIENT_SUFFIX} blenlib${CLIENT_SUFFIX})
endif()

if(WITH_CLIENT_ULTRAGRID_LIB)
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${ULTRAGRID_LIBRARIES})
endif()

if(WITH_NVPIPE)
	#${CUDA_LIBRARIES} ${CUDA_LIB}
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${NVPIPE_LIBRARIES} ${CUDA_CUSTOM_LIBRARIES})
endif()

if(WITH_CLIENT_GPUJPEG)
	target_link_libraries(blender_client${CLIENT_SUFFIX} ${GPUJPEG_LIBRARIES})
endif()

install (TARGETS blender_client${CLIENT_SUFFIX} DESTINATION bin)
