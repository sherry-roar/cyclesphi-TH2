set(INC
	.
	../../intern/cycles
    ../../intern/cycles/kernel
	../../intern/cycles/kernel/device/client
	../../intern/cycles/kernel/device/cuda
	../api
	../../intern/atomic
	../../intern/cycles/util
)

set(SRC
	${INC}
    ../../intern/cycles/kernel/device/cuda/kernel.cu
)

add_definitions(-DBLENDER_CLIENT)
add_definitions(-DWITH_CLIENT_CUDA)

if(WITH_CUDA_CPUIMAGE)
	add_definitions(-DWITH_CUDA_CPUIMAGE)
endif()

#include_directories(${INC})

# CUDA module
	# 64 bit only
	set(CUDA_BITS 64)

	# CUDA version
	execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
	string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR "${NVCC_OUT}")
	string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${NVCC_OUT}")
	set(CUDA_VERSION "${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}")

	# build for each arch
	set(cuda_cubins)
    
    # pair of macros to allow libraries to be specify files to install, but to
    # only install them at the end so the directories don't get cleared with
    # the files in them. used by cycles to install addon.
    function(delayed_install
    	base
    	files
    	destination)
    
    	foreach(f ${files})
    		if(IS_ABSOLUTE ${f})
    			set_property(GLOBAL APPEND PROPERTY DELAYED_INSTALL_FILES ${f})
    		else()
    			set_property(GLOBAL APPEND PROPERTY DELAYED_INSTALL_FILES ${base}/${f})
    		endif()
    		set_property(GLOBAL APPEND PROPERTY DELAYED_INSTALL_DESTINATIONS ${destination})
    	endforeach()
    endfunction()    

	macro(CUDA_KERNEL_ADD arch prev_arch name flags sources stat)
		set(cuda_cubin ${name}_${arch}.cubin)
		if(${stat} STREQUAL "stat")
			set(cuda_cubin ${name}_${arch}_stat.cubin)
		endif()

		set(kernel_sources ${sources})
		if(NOT ${prev_arch} STREQUAL "none")
			set(kernel_sources ${kernel_sources} ${name}_${prev_arch}.cubin)
		endif()

		set(cuda_kernel_src "../../intern/cycles/kernel/device/cuda/${name}.cu")

		set(cuda_flags
			-D CCL_NAMESPACE_BEGIN=
			-D CCL_NAMESPACE_END=
			-D NVCC
			-D BLENDER_CLIENT
			-m ${CUDA_BITS}
			-I ${CMAKE_CURRENT_SOURCE_DIR}/../../intern/cycles
			-I ${CMAKE_CURRENT_SOURCE_DIR}/../../intern/cycles/kernel/device/cuda
			--use_fast_math
			-o ${CMAKE_CURRENT_BINARY_DIR}/${cuda_cubin}
		)
      #-g
      #-G

     if(WITH_CUDA_DEBUG)
			list(APPEND cuda_flags
			-g
			-G
			)
		endif()
		if(WITH_CUDA_DEBUG_LINEINFO)
			list(APPEND cuda_flags
				-lineinfo
			)
		endif()

		if(WITH_NANOVDB)
			list(APPEND cuda_flags
			-DWITH_NANOVDB
			)

			list(APPEND cuda_flags
					-I"${NANOVDB_INCLUDE_DIR}"
					)
		endif()

		if(WITH_CUDA_CPUIMAGE)
				list(APPEND cuda_flags
					-DWITH_CUDA_CPUIMAGE
				)
		endif()

		if(WITH_CUDA_SRGB)
				list(APPEND cuda_flags
					-DWITH_CUDA_SRGB
				)
		endif()

		if(WITH_CLIENT_CUDA_GPU_TILES)
			list(APPEND cuda_flags
				-DWITH_CLIENT_CUDA_GPU_TILES
			)
		endif()

		if(WITH_VRCLIENT)
				list(APPEND cuda_flags
					-DWITH_VRCLIENT
				)
		endif()	
			
		if(WITH_OPTIX_DENOISER)
				list(APPEND cuda_flags
					-DWITH_OPTIX_DENOISER
				)
		endif()

		if(WITH_ORIG_TILES)
				list(APPEND cuda_flags
					-DWITH_ORIG_TILES
				)
		endif()

		if(WITH_CLIENT_MPI)
			list(APPEND cuda_flags
					-DWITH_CLIENT_MPI
					)
		endif()		

		if(WITH_CLIENT_MPI_SOCKET)
			list(APPEND cuda_flags
					-DWITH_CLIENT_MPI_SOCKET
					)
		endif()

		if(WITH_CLIENT_SOCKET)
			list(APPEND cuda_flags
					-DWITH_CLIENT_SOCKET
					)
		endif()

		if(ENABLE_LOAD_BALANCEv2)
				list(APPEND cuda_flags
					-DENABLE_LOAD_BALANCEv2
				)
		endif()

        if(WITH_CLIENT_YUV)
			list(APPEND cuda_flags
					-DWITH_CLIENT_YUV
					)
		endif()

		if(WITH_CLIENT_ULTRAGRID_LIB)
			list(APPEND cuda_flags
					-DWITH_CLIENT_ULTRAGRID_LIB
					)
		endif()

		if(WITH_RGBA_FORMAT)
			list(APPEND cuda_flags
					-DWITH_RGBA_FORMAT
					)
		endif()

		if(WITH_SIZE_UCHAR3)
			list(APPEND cuda_flags
					-DWITH_SIZE_UCHAR3
					)
		endif()

		if(WITH_CLIENT_CACHE)
			list(APPEND cuda_flags
					-DWITH_CLIENT_CACHE
					)
		endif()

		if(${stat} STREQUAL "stat")
				list(APPEND cuda_flags
					-D__KERNEL_CUDA_STAT__
				)
		endif()

		if(WITH_CLIENT_SHOW_STAT)
				list(APPEND cuda_flags
					-DWITH_CLIENT_SHOW_STAT
				)
		endif()    

		add_custom_command(
			OUTPUT ${cuda_cubin}
			COMMAND ${CUDA_NVCC_EXECUTABLE}
					-arch=${arch}					
					${CUDA_NVCC_FLAGS}
					--cubin
					${CMAKE_CURRENT_SOURCE_DIR}/${cuda_kernel_src}
				${cuda_flags}
			DEPENDS ${kernel_sources})

		delayed_install("${CMAKE_CURRENT_BINARY_DIR}" "${cuda_cubin}" lib)
		list(APPEND cuda_cubins ${cuda_cubin})

		unset(cuda_debug_flags)
	endmacro()

	set(prev_arch "none")
	foreach(arch ${CUDA_BINARIES_ARCH})
		# Compile regular kernel
		CUDA_KERNEL_ADD(${arch} ${prev_arch} kernel "" "${SRC}" "none")
		if(WITH_CUDA_STAT)
			CUDA_KERNEL_ADD(${arch} ${prev_arch} kernel "" "${SRC}" "stat")
		endif()
	endforeach()

add_custom_target(cycles_kernel_cuda ALL DEPENDS ${cuda_cubins})
