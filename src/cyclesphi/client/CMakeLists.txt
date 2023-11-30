cmake_minimum_required(VERSION 3.1)

project (blender_client)

#==============================CYCLESPHI==============================
add_definitions(
    -DCCL_NAMESPACE_BEGIN=
    -DCCL_NAMESPACE_END=
)

option(WITH_CLIENT_CPU          "Enable CPU/OMP" ON)
option(WITH_CPU_SIMD             "Enable SSE (has to be supported by the compiler)" ON)
option(WITH_CPU_AVX             "Enable AVX (has to be supported by the compiler)" ON)
option(WITH_CPU_AVX2            "Enable AVX2 (has to be supported by the compiler)" ON)
option(WITH_IMAGE_TURBOJPEG     "Enable TurboJPEG" OFF)
option(WITH_IMAGE_X264          "Enable X264" OFF)
option(WITH_NVPIPE              "Enable NVPIPE" OFF)
option(WITH_CLIENT_GPUJPEG              "Enable GPUJPEG" OFF)

option(WITH_RJSON              "Enable RJSON" OFF)

#option(WITH_RGBA_HALF              "Enable RGBA_HALF" OFF)

option(WITH_KNC_NATIVE          "Enable build for KNC" OFF)
option(WITH_KNC_NATIVE_GCC      "Enable build for KNC" OFF)

option(WITH_CLIENT_CUDA         "Enable build for CUDA" OFF)
option(WITH_CUDA_STAT           "Enable statistics for CUDA BIN" OFF)
option(WITH_CUDA_STATv2           "Enable statistics for CUDA BIN" OFF)
option(WITH_CUDA_STATv2_LB           "Enable statistics for CUDA BIN" OFF)
option(WITH_CUDA_DEBUG           "Enable debug for CUDA" OFF)
option(WITH_CUDA_DEBUG_LINEINFO           "Enable debug for CUDA" OFF)

option(WITH_CLIENT_OPTIX         "Enable build for OPTIX" OFF)

option(WITH_OPTIX_DENOISER      "Enable build for OPTIX" OFF)
option(WITH_OIDN_DENOISER      "Enable build for OIDN" OFF)

option(WITH_CLIENT_HIP         "Enable build for HIP" OFF)
option(WITH_HIP_STAT           "Enable statistics for HIP" OFF)
option(WITH_HIP_STATv2           "Enable statistics for HIP" OFF)
option(WITH_HIP_STATv2_LB           "Enable statistics for HIP" OFF)


option(WITH_CLIENT_SHOW_STAT         "Enable build for SHOW STAT" OFF)
option(WITH_CLIENT_SHOW_STAT_BVH_LOOP         "Enable build for SHOW STAT" OFF)

option(WITH_CLIENT_CUDA_CPU_STAT         "Enable build for CUDA CPU STAT" OFF)
option(WITH_CLIENT_CUDA_CPU_STAT_LB         "Enable build for CUDA CPU STAT" OFF)
option(WITH_CLIENT_CUDA_CPU_STAT2         "Enable build for CUDA CPU STAT" OFF)
option(WITH_CLIENT_CUDA_CPU_STAT2v2         "Enable build for CUDA CPU STAT" OFF)
option(WITH_CLIENT_CUDA_CPU_STAT2_LB         "Enable build for CUDA CPU STAT" OFF)
option(WITH_CLIENT_CUDA_CPU_STAT3         "Enable build for CUDA CPU STAT" OFF)

#############

option(UNITEST1_ROUND_ROBIN "Enable build for CUDA RR UNIMEM" OFF)
option(UNITEST1_ROUND_ROBIN2 "Enable build for CUDA RR UNIMEM" OFF)

option(UNITEST1_RANDOM "Enable build for CUDA RANDOM UNIMEM" OFF)
option(UNITEST1_COUNTINOUS "Enable build for CUDA CON UNIMEM" OFF)

option(UNITEST1_CPU "Enable build for CUDA CPU UNIMEM" OFF)
option(UNITEST1_CREDITS_FILE "Enable build for CUDA FILE STAT" OFF)

option(UNITEST2_ROUND_ROBIN "Enable build for CUDA RR STAT" OFF)
option(UNITEST2_RANDOM "Enable build for CUDA RANDOM STAT" OFF)

option(UNITEST2_CREDITS "Enable build for CUDA CREDIT STAT" OFF)
option(UNITEST2_CREDITS_CPU "Enable build for CUDA CREDIT CPU STAT" OFF)
option(UNITEST2_CREDITS_CPU2 "Enable build for CUDA CREDIT CPU STAT" OFF)

#############

#option(WITH_CLIENT_GPU_PER_MPI_RANK         "Enable build for CUDA_PER_MPI_RANK" OFF)

option(WITH_SOCKET_UDP          "Enable UDP" OFF)
option(WITH_SOCKET_ONLY_DATA          "Enable SOCKET_ONLY_DATA" OFF)
option(WITH_CLIENT_MPI_SOCKET   "Enable MPI_SOCKET" OFF)
option(WITH_CLIENT_NCCL_SOCKET              "Enable NCCL_SOCKET" OFF)
option(WITH_CLIENT_MPI_REDUCE   "Enable MPI_REDUCE" OFF)
option(WITH_CLIENT_SOCKET   "Enable SOCKET" OFF)
option(WITH_CLIENT_MPI          "Enable MPI" OFF)
option(WITH_CLIENT_FILE         "Enable FILE" OFF)
option(WITH_CLIENT_FILE_MMAP    "Enable FILE" OFF)
option(WITH_CLIENT_MPI_FILE     "Enable FILE+MPI" OFF)

option(WITH_CLIENT_CACHE    "Enable CACHE_IMAGE" OFF)
option(WITH_CLIENT_MPI_CACHE    "Enable CACHE_IMAGE" OFF)

#==============================RE==============================
option(WITH_CLIENT_RENDERENGINE "Enable RENDERENGINE" OFF)
option(WITH_CLIENT_RENDERENGINE_ADDON "Enable RENDERENGINE_ADDON" OFF)
option(WITH_CLIENT_RENDERENGINE_VR "Enable RENDERENGINE_VR" OFF)
#option(WITH_CLIENT_RENDERENGINE_VRCLIENT "Enable RENDERENGINE_VRCLIENT" OFF)
option(WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB "Enable RENDERENGINE_ULTRAGRID" OFF)
option(WITH_CLIENT_RENDERENGINE_EMULATE "Enable RENDERENGINE" OFF)
option(WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD "Enable RENDERENGINE" OFF)
option(WITH_CLIENT_RENDERENGINE_SENDER     "Enable build for RENDERENGINE_SENDER" OFF)
#============================================================

option(WITH_CLIENT_UNIMEM       "Enable build for UNIMEM" OFF)
option(ENABLE_LOAD_BALANCE      "Enable ENABLE_LOAD_BALANCE - line stealing" OFF)
option(ENABLE_LOAD_BALANCE_EXIT "Enable ENABLE_LOAD_BALANCE_EXIT" OFF)
option(ENABLE_LOAD_BALANCE_CUDA      "Enable ENABLE_LOAD_BALANCE_CUDA - faster lines+time stealing" OFF)
option(ENABLE_LOAD_BALANCEv2    "Enable ENABLE_LOAD_BALANCEv2 - pixel stealing" OFF)
option(ENABLE_LOAD_BALANCEv3    "Enable ENABLE_LOAD_BALANCEv3 - faster lines+time stealing" OFF)

option(CLIENT_MPI_LOAD_BALANCING_SAMPLES "Enable LOAD_BALANCING_SAMPLES" OFF)
option(CLIENT_MPI_LOAD_BALANCING_LINES "Enable LINE_LOAD_BALANCING" OFF)

option(WITH_LOAD_BALANCING_COLUMN "Enable COLUMN LB, default: ROW LB" OFF)

option(ENABLE_INC_SAMPLES       "Enable ENABLE_INC_SAMPLES" OFF)
option(ENABLE_STEP_SAMPLES       "Enable ENABLE_STEP_SAMPLES" OFF)

option(WITH_CUDA_CPUIMAGE           "Enable CPU_IMAGE" OFF)
option(WITH_HIP_CPUIMAGE           "Enable CPU_IMAGE" OFF)
option(WITH_NANOVDB              "Enable NANO" OFF)
option(WITH_CPU_STAT            "Enable statistics for CPU" OFF)

option(WITH_CUDA_SRGB           "Enable CUDA_SRGB" OFF)
option(WITH_CLIENT_FORCE_CUDA "Enable FORCE CUDA OVER OPTIX" OFF)

option(WITH_WEBRTC            "Enable WEBRTC" OFF)
option(WITH_ORIG_TILES           "Enable ORIG_TILES" OFF)

option(WITH_CLIENT_CUDA_GPU_TILES           "Enable CLIENT_CUDA_GPU_TILES" OFF)

#==============================POP==============================
option(WITH_POP "Enable" OFF)
option(CLIENT_PATH_TRACE_ROW "Enable" OFF)
option(CLIENT_PATH_TRACE_COLUMN "Enable" OFF)
option(CLIENT_PATH_TRACE_TILE "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_1 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_2 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_4 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_8 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_16 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_32 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_64 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_128 "Enable" OFF)
option(CLIENT_PATH_TRACE_PIXEL_256 "Enable" OFF)

option(WITH_BUFFER_DECODER "Buffer Decoder" OFF)
option(WITH_CLIENT_BUFFER_DECODER "Buffer Decoder" OFF)

#==============================ULTRAGRID==============================
option(WITH_NVML "Build WITH_NVML" OFF)

#==============================ULTRAGRID==============================
option(WITH_CLIENT_VRGSTREAM     "Enable build for VRGSTREAM" OFF)
option(WITH_CLIENT_VRRECEIVER     "Enable build for VRRECEIVER" OFF)
option(WITH_CLIENT_XOR_RLE     "Enable build for XOR_RLE" OFF)
option(WITH_CLIENT_YUV         "Enable build for YUV" OFF)
option(WITH_SIZE_UCHAR3         "Enable build for UCHAR3" OFF)
option(WITH_CUDA_GL_INTEROP        "Enable build for CUDA_GL_INTEROP" OFF)
option(WITH_CLIENT_ULTRAGRID     "Enable build for ULTRAGRID" OFF)
option(WITH_CLIENT_ULTRAGRID_LIB     "Enable build for ULTRAGRID_LIB" OFF)
option(WITH_RGBA_FORMAT		  "Enable rgba format, default is yuv420" OFF)
option(WITH_YUV_OUTPUT        "Enable yuv output - writing to yuv file" OFF)
option(WITH_BMP_OUTPUT        "Enable bmp output - writing to bmp file" OFF)
option(WITH_CUDA_BUFFER_MANAGED         "Enable build for CUDA_BUFFER_MANAGED" OFF)

option(WITH_VRCLIENT     "Enable build for VRCLIENT" OFF)
option(WITH_VRCLIENT_VRGSTREAM     "Enable build for VRCLIENT_VRGSTREAM" OFF)
option(WITH_VRCLIENT_ONLY     "Enable build for VRCLIENT_VRGSTREAM" OFF)

#==============================VRCLIENT==============================

option(WITH_VRCLIENT     "Enable build for VRCLIENT" OFF)
option(WITH_OPENXR_SIMPLE     "Enable build for OPENXR" OFF)
option(WITH_VRCLIENT_OPENVR     "Enable build for OPENVR" OFF)
option(WITH_VRCLIENT_GLSTEREO     "Enable build for GLSTEREO" OFF)
option(WITH_VRCLIENT_RECEIVER     "Enable build for RECEIVER" OFF)
option(WITH_VRCLIENT_XTAL     "Enable build for XTAL" OFF)

option(WITH_CLIENT_MPI_VRCLIENT              "Enable MPI_VRCLIENT" OFF)

#============================================================
option(WITH_NETREFLECTOR     "Enable build for NETREFLECTOR" OFF)
#============================================================

set(WITH_OPENMP ON)
set(CLIENT_SUFFIX "")
set(CMAKE_CXX_FLAGS "-fopenmp -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build_files/cmake/Modules")

if(WITH_CLIENT_CUDA_GPU_TILES)
  set(ENABLE_LOAD_BALANCE      OFF)
  set(ENABLE_LOAD_BALANCE_EXIT OFF)
  set(ENABLE_LOAD_BALANCE_CUDA OFF)
  set(ENABLE_LOAD_BALANCEv2    OFF)
  set(ENABLE_LOAD_BALANCEv3    OFF)
endif()

if(WITH_CLIENT_MPI)
  set(WITH_CLIENT_GPUJPEG OFF)
endif()

if(WITH_CLIENT_ULTRAGRID_LIB)
    set(WITH_CLIENT_ULTRAGRID ON)
endif()

if(WITH_CUDA_STATv2)
    set(WITH_CUDA_STAT ON)
endif()

if(WITH_HIP_STATv2)
    set(WITH_HIP_STAT ON)
endif()

if(WITH_CLIENT_MPI_SOCKET)
    set(WITH_CLIENT_MPI OFF)
endif()

#if(WITH_CLIENT_CUDA)
#    set(WITH_CUDA_CPUIMAGE ON)
#endif()

if(WITH_CLIENT_CUDA_CPU_STAT3)
    set(WITH_CLIENT_CUDA_CPU_STAT ON)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT2)
    set(WITH_CLIENT_CUDA_CPU_STAT ON)
endif()

if(WITH_CLIENT_CUDA_CPU_STAT)
	set(WITH_CLIENT_CUDA ON)
  set(WITH_CLIENT_CPU ON)
  set(WITH_CPU_STAT ON)
  set(WITH_CUDA_STAT OFF)
  set(WITH_CLIENT_UNIMEM ON)

  set(WITH_CPU_SIMD OFF)
	set(WITH_CPU_AVX OFF)
	set(WITH_CPU_AVX2 OFF)

    #set(WITH_CUDA_CPUIMAGE ON)

    #set(WITH_CLIENT_CUDA_CPU_STAT2 ON)
    #set(WITH_CLIENT_CUDA_CPU_STAT_LB ON)
    #set(WITH_CLIENT_CUDA_CPU_STAT2_LB ON)

endif()

if(WITH_CUDA_STAT)
  set(WITH_CLIENT_CUDA ON)
  set(WITH_CLIENT_CPU OFF)
  set(WITH_CPU_STAT OFF)
  set(WITH_CLIENT_CUDA_CPU_STAT OFF)
  set(WITH_CLIENT_CUDA_CPU_STAT2 OFF)
  set(WITH_CLIENT_UNIMEM ON)
endif()

if(WITH_HIP_STAT)
  set(WITH_CLIENT_HIP ON)
  set(WITH_CLIENT_CPU OFF)
  set(WITH_CPU_STAT OFF)
  set(WITH_CLIENT_UNIMEM ON)
endif()

if(WITH_CLIENT_SHOW_STAT)
  set(ENABLE_INC_SAMPLES OFF)
endif()

if(ENABLE_LOAD_BALANCEv3)
  set(ENABLE_LOAD_BALANCE ON)
endif()

if(WITH_CLIENT_FILE AND NOT WIN32)
  set(WITH_CLIENT_FILE_MMAP ON)
endif()

#message(${CMAKE_CXX_COMPILER_ID})

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
 #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-vec -no-inline-max-total-size -no-inline-max-per-compile -inline-factor=150 -fp-model precise -prec-div -prec-sqrt -parallel")

		if(WITH_KNC_NATIVE)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
			set(CLIENT_SUFFIX "-mic")
			set(WITH_CPU_SIMD OFF)
			set(WITH_CPU_AVX OFF)
			set(WITH_CPU_AVX2 OFF)
		elseif(WITH_CPU_AVX2)
			set(WITH_CPU_AVX ON)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xcore-avx2")
		elseif(WITH_CPU_AVX)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xavx")
		endif()
elseif(CMAKE_COMPILER_IS_GNUCC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -ldl -fpermissive -fno-trapping-math -fno-math-errno -fno-rounding-math -fno-signaling-nans -mfpmath=sse -msse -msse2 -msse3 -mssse3 -msse4.1")
    if(WITH_KNC_NATIVE_GCC)
		set(CMAKE_CXX_FLAGS "-fopenmp -std=c++11 -fpermissive -fno-trapping-math -fno-math-errno -fno-rounding-math -fno-signaling-nans -L/apps/all/impi/2017.4.239-iccifort-2017.7.259-GCC-6.3.0-2.27/mic/lib")
		set(CLIENT_SUFFIX "-mic")
		set(WITH_CPU_SIMD OFF)
		set(WITH_CPU_AVX OFF)
		set(WITH_CPU_AVX2 OFF)
	elseif(WITH_CPU_AVX2)
		set(WITH_CPU_AVX ON)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -mlzcnt -mbmi -mbmi2 -mf16c")
	elseif(WITH_CPU_AVX)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
	endif()
elseif(WIN32 AND MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "/openmp /std:c++latest")
  set(CMAKE_CXX_FLAGS_RELEASE "")
  set(CMAKE_CXX_FLAGS_DEBUG "")
	#set(WITH_CPU_SIMD ON)
	#set(WITH_CPU_AVX ON)
	#set(WITH_CPU_AVX2 ON)

  # /arch:AVX for VC2012 and above
  if(NOT MSVC_VERSION LESS 1700)
    set(CYCLES_AVX_ARCH_FLAGS "/arch:AVX")
    set(CYCLES_AVX2_ARCH_FLAGS "/arch:AVX /arch:AVX2")
  elseif(NOT CMAKE_CL_64)
    set(CYCLES_AVX_ARCH_FLAGS "/arch:SSE2")
    set(CYCLES_AVX2_ARCH_FLAGS "/arch:SSE2")
  endif()

  # Unlike GCC/clang we still use fast math, because there is no fine
  # grained control and the speedup we get here is too big to ignore.
  set(CYCLES_KERNEL_FLAGS "/fp:fast -D_CRT_SECURE_NO_WARNINGS /GS-")

  # there is no /arch:SSE3, but intrinsics are available anyway
  if(CMAKE_CL_64)
    set(CYCLES_SSE2_KERNEL_FLAGS "${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_SSE3_KERNEL_FLAGS "${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_SSE41_KERNEL_FLAGS "${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_AVX_KERNEL_FLAGS "${CYCLES_AVX_ARCH_FLAGS} ${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_AVX2_KERNEL_FLAGS "${CYCLES_AVX2_ARCH_FLAGS} ${CYCLES_KERNEL_FLAGS}")
  else()
    set(CYCLES_SSE2_KERNEL_FLAGS "/arch:SSE2 ${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_SSE3_KERNEL_FLAGS "/arch:SSE2 ${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_SSE41_KERNEL_FLAGS "/arch:SSE2 ${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_AVX_KERNEL_FLAGS "${CYCLES_AVX_ARCH_FLAGS} ${CYCLES_KERNEL_FLAGS}")
    set(CYCLES_AVX2_KERNEL_FLAGS "${CYCLES_AVX2_ARCH_FLAGS} ${CYCLES_KERNEL_FLAGS}")
  endif()
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CYCLES_KERNEL_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox")
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG /D PLATFORM_WINDOWS /MTd /Zi /Ob0 /Od /RTC1 /DPSAPI_VERSION=1 /DOIIO_STATIC_BUILD /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG /D PLATFORM_WINDOWS /MDd /Zi /Ob0 /Od /RTC1 /DPSAPI_VERSION=1 /DOIIO_STATIC_BUILD /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Ox")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /Ox")
endif()

if(WITH_CLIENT_ULTRAGRID_LIB)
    set(ULTRAGRID_INCLUDE_DIR "" CACHE PATH "")
    set(ULTRAGRID_LIBRARIES "" CACHE FILEPATH "")
endif()

if(WITH_IMAGE_TURBOJPEG)
    set(TURBOJPEG_INCLUDE_DIR "" CACHE PATH "")
    set(TURBOJPEG_LIBRARIES "" CACHE FILEPATH "")
endif()

if(WITH_IMAGE_X264)
    set(X264_INCLUDE_DIR "" CACHE PATH "")
    set(X264_LIBRARIES "" CACHE FILEPATH "")
endif()

if(WITH_WEBRTC)
    set(WEBRTC_INCLUDE_DIR "" CACHE PATH "")
    set(WEBRTC_LIBRARIES "" CACHE FILEPATH "")
endif()

if(WITH_CLIENT_MPI)
  set(CLIENT_MPI_LOAD_BALANCING_SAMPLES ON)
endif()

if(WITH_CLIENT_NCCL_SOCKET)
  set(WITH_CLIENT_MPI_SOCKET ON)
endif()

if(WITH_CLIENT_MPI AND NOT WITH_KNC_NATIVE_GCC AND NOT WITH_CLIENT_MPI_GCC)
    find_package(MPI)
endif()

if(WITH_CLIENT_MPI_SOCKET AND NOT WITH_CLIENT_MPI_GCC)
	find_package(MPI)
endif()

if(WITH_CLIENT_MPI_FILE AND NOT WITH_CLIENT_MPI_GCC)
	find_package(MPI)
endif()

if(WITH_CLIENT_NCCL_SOCKET)  
  #find_package(NCCL REQUIRED)
	set(NCCL_INCLUDE_DIR "" CACHE PATH "")
  set(NCCL_LIBRARIES "" CACHE FILEPATH "")  
endif()

if(WITH_OPTIX_DENOISER OR WITH_CLIENT_OPTIX)
  find_package(OptiX)
endif()

if (WITH_CLIENT_CUDA OR WITH_CLIENT_OPTIX)
  find_package(CUDA REQUIRED)
	set(CUDA_BINARIES_ARCH sm_30 sm_35 sm_37 sm_50 sm_52 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 CACHE STRING "CUDA architectures to build binaries for")
	set(CUDA_CUSTOM_LIBRARIES "" CACHE FILEPATH "")
endif()

if(WITH_CUDA_GL_INTEROP)
    find_package(CUDA REQUIRED)
    set(CUDA_CUSTOM_LIBRARIES "" CACHE FILEPATH "")
endif()

if(WITH_NVPIPE)
	set(NVPIPE_INCLUDE_DIR "" CACHE PATH "")
    set(NVPIPE_LIBRARIES "" CACHE FILEPATH "")
	find_package(CUDA REQUIRED)

	# Construct path to CUDA driver API lib (not provided by FindCUDA)
	#get_filename_component(CUDA_LIB_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
	#find_library(CUDA_LIB NAMES cuda HINTS ${CUDA_LIB_DIR})
endif()

if(WITH_CLIENT_GPUJPEG)
	set(GPUJPEG_INCLUDE_DIR "" CACHE PATH "")
  set(GPUJPEG_LIBRARIES "" CACHE FILEPATH "")
endif()

# Subdirectories
if (WITH_CLIENT_CUDA)
    add_subdirectory(cycles_cuda)
endif()

if(WITH_CUDA_DEBUG_LINEINFO)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
endif()

if (WITH_CLIENT_OPTIX)
    add_subdirectory(cycles_optix)
    add_subdirectory(cycles_cuda)
endif()

if (WITH_CLIENT_HIP)
    add_subdirectory(hipew)
    add_subdirectory(cycles_hip)
endif()

if (WITH_CLIENT_RENDERENGINE_ADDON)
    #add_subdirectory(glew)
    add_subdirectory(glew-mx)
    set(GLEW_INCLUDE_DIR "" CACHE PATH "")
    set(GLEW_LIBRARIES "" CACHE FILEPATH "")
    add_subdirectory(renderengine)

endif()

if (WITH_CLIENT_VRGSTREAM)
    add_subdirectory(vrgstream)
endif()

if (WITH_NETREFLECTOR)
    add_subdirectory(netreflector)
endif()

if (WITH_CLIENT_VRRECEIVER)
	  set(VRGSTREAM_INCLUDE_DIR "" CACHE PATH "")
    set(VRGSTREAM_LIBRARIES "" CACHE FILEPATH "")

    add_subdirectory(vrreceiver)
endif() 

if (WITH_CLIENT_ULTRAGRID)
    # Find CUDA
    find_package(CUDA)
    #if (NOT WITH_CLIENT_ULTRAGRID_LIB)
    add_subdirectory(ultragrid)
    #endif()
endif()

if (WITH_CLIENT_CUDA OR WITH_CLIENT_CPU OR WITH_CLIENT_HIP OR WITH_CLIENT_OPTIX)
  add_subdirectory(cycles)
  add_subdirectory(blender_client)
endif()

add_subdirectory(blenlib)

if(WITH_VRCLIENT_OPENVR)
    set(OPENVR_INCLUDE_DIR "" CACHE PATH "")
    set(OPENVR_LIBRARIES "" CACHE FILEPATH "")
endif()

if (WITH_VRCLIENT)
  set(WITH_CUDA_GL_INTEROP ON)
  set(CMAKE_MODULE_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  )                 
  
  find_package(GLFW REQUIRED)
  
  # Subdirectories
  add_subdirectory(glad)
  add_subdirectory(vrclient)

#  if (NOT WITH_CLIENT_CUDA OR NOT WITH_CLIENT_CPU)
#    add_subdirectory(blenlib)
#  endif()

endif()

if(WITH_NANOVDB)
    set(NANOVDB "" CACHE FILEPATH "")
    set(NANOVDB_INCLUDE_DIR  "" CACHE PATH "")
endif()

if(WITH_OPENXR_SIMPLE)
    set(SDL2_INCLUDE_DIR "" CACHE PATH "")
    set(SDL2_LIBRARIES "" CACHE FILEPATH "")

    set(OPENXR_INCLUDE_DIR "" CACHE PATH "")
    set(OPENXR_LIBRARIES "" CACHE FILEPATH "")

  add_subdirectory(openxr_simple)
endif()

if(WITH_BUFFER_DECODER)
  set(BLEND_LIB_DIR "" CACHE PATH "")
  if(WIN32)
		set(ZLIB_INCLUDE_DIR ${BLEND_LIB_DIR}/zlib/include)
		set(ZLIB_STATIC_LIB ${BLEND_LIB_DIR}/zlib/lib/libz_st.lib)

		set(PNG_INCLUDE_DIR ${BLEND_LIB_DIR}/png/include)
		set(PNG_STATIC_LIB ${BLEND_LIB_DIR}/png/lib/libpng.lib)
  else()
		set(ZLIB_INCLUDE_DIR "" CACHE PATH "")
		set(ZLIB_STATIC_LIB "" CACHE FILEPATH "")

		set(PNG_INCLUDE_DIR "" CACHE PATH "")
		set(PNG_STATIC_LIB "" CACHE FILEPATH "")
  endif()
  if(WITH_RJSON)
	find_package(RapidJSON REQUIRED)
  endif()
  add_subdirectory(buffer_decoder)
endif()
