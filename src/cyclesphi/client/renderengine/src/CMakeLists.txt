set(INC
	 .
   ../../api
   ../../../intern/cycles/kernel/device/client
   ../../ultragrid
   #../../../extern/glew/include
   ../../../intern/glew-mx

  ../../../source/blender/blenlib
  ../../../source/blender/makesdna
 
  ${NVPIPE_INCLUDE_DIR}
	${CUDA_TOOLKIT_INCLUDE}
	${CUDA_INCLUDE_DIRS}
  ${ULTRAGRID_INCLUDE_DIR}

  ${GLEW_INCLUDE_DIR}
)

set(SRC
	cyclesphi.cpp
  cyclesphi.h
)

set(SRC_VRG
  cyclesphi_vrg.cpp
  cyclesphi.h
  ../../ultragrid/vrgstream.h
)

set(SRC_HEADERS
    cyclesphi_api.h
    cyclesphi_data.h

    ../../cycles/kernel_util.cpp
    ../../cycles/kernel_util.h
    ../../api/client_api.h
)

if(WITH_OPENMP)
	add_definitions(-DWITH_OPENMP)
endif()

if(WITH_SOCKET_UDP)
	add_definitions(-DWITH_SOCKET_UDP)
endif()

if(WITH_SOCKET_ONLY_DATA)
	add_definitions(-DWITH_SOCKET_ONLY_DATA)
endif()

if(WITH_CUDA_GL_INTEROP)
	add_definitions(-DWITH_CUDA_GL_INTEROP)
  add_definitions(-DWITH_OPENGL)

  add_definitions(-DWITH_LEGACY_OPENGL)
  add_definitions(-DGLEW_STATIC)
  add_definitions(-DGLEW_NO_ES)
  add_definitions(-DGLEW_NO_GLU)
endif()

if(WITH_OPTIX_DENOISER)
	add_definitions(-DWITH_OPTIX_DENOISER)
endif()

if(WITH_NVPIPE)
	add_definitions(-DWITH_NVPIPE)
endif()

if(WITH_CLIENT_ULTRAGRID AND NOT WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB)
	  add_definitions(-DWITH_CLIENT_ULTRAGRID)
    
    list(APPEND SRC
      ../../ultragrid/platform_ipc.c
      ../../../intern/cycles/kernel/device/client/kernel_tcp.cpp
    )

    list(APPEND SRC_HEADERS
        ../../ultragrid/platform_ipc.h
        ../../../intern/cycles/kernel/device/client/kernel_tcp.h
    )
elseif(WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB)
	  add_definitions(-DWITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB)
    list(APPEND SRC
      ../../ultragrid/vrgstream.h
    )
else()
    list(APPEND SRC
        ../../../intern/cycles/kernel/device/client/kernel_tcp.cpp
    )

    list(APPEND SRC_HEADERS
        ../../../intern/cycles/kernel/device/client/kernel_tcp.h
    )
endif()

if(WITH_CLIENT_RENDERENGINE)
	add_definitions(-DWITH_CLIENT_RENDERENGINE)
endif()

if(WITH_CLIENT_RENDERENGINE_VR)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_VR)
endif()

if(WITH_CLIENT_RENDERENGINE_SENDER)
	add_definitions(-DWITH_CLIENT_RENDERENGINE_SENDER)
endif()

add_definitions(
	-DCCL_NAMESPACE_BEGIN=namespace\ ccl\ {
	-DCCL_NAMESPACE_END=}
)

#message("GLEW_INCLUDE_DIR")
#message("${GLEW_INCLUDE_DIR}")

add_definitions(-DWITH_CLIENT_RENDERENGINE)

include_directories(${INC})
add_library(cyclesphi_renderengine SHARED ${SRC} ${SRC_HEADERS})

target_link_libraries(cyclesphi_renderengine blenlib)

	if(WITH_CUDA_GL_INTEROP)
    #target_link_libraries(cyclesphi_renderengine extern_glew)
    target_link_libraries(cyclesphi_renderengine glew_mx)
    target_link_libraries(cyclesphi_renderengine ${GLEW_LIBRARIES})
		target_link_libraries(cyclesphi_renderengine ${CUDA_CUSTOM_LIBRARIES})
	endif()

if(WITH_NVPIPE)
	#${CUDA_LIBRARIES} ${CUDA_LIB}
	target_link_libraries(cyclesphi_renderengine ${NVPIPE_LIBRARIES} ${CUDA_CUSTOM_LIBRARIES})
endif()

if(WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB)
  target_link_libraries(cyclesphi_renderengine ${ULTRAGRID_LIBRARIES})

	add_library(vrgstream SHARED ${SRC_VRG})
	set_target_properties(vrgstream PROPERTIES PUBLIC_HEADER "../../ultragrid/vrgstream.h")
  target_link_libraries(vrgstream cyclesphi_renderengine)

	install (TARGETS vrgstream
			LIBRARY DESTINATION lib
			PUBLIC_HEADER DESTINATION include
			)
endif()

#install (TARGETS cyclesphi_renderengine DESTINATION lib)

install (TARGETS cyclesphi_renderengine
			LIBRARY DESTINATION lib
			)
