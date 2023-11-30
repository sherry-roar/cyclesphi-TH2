set(INC
	.
	../../extern/Eigen3
    ../../source/blender/blenlib
	../../source/blender/makesdna
)

set(SRC
     math_matrix.cpp
     math_rotation.cpp
     math_vector.cpp
     math_vector_inline.cpp
		 math_base.cpp
		 math_base_inline.cpp
)

set(SRC_HEADERS
)

if(WITH_OPENMP)
	add_definitions(-DWITH_OPENMP)
endif()

add_definitions(-D__SSE2__)
add_definitions(-DNDEBUG)
add_definitions(-DMATH_STANDALONE)

include_directories(${INC})

if(WIN32)
	add_library(blenlib${CLIENT_SUFFIX} STATIC ${SRC} ${SRC_HEADERS})
else()
	add_library(blenlib${CLIENT_SUFFIX} SHARED ${SRC} ${SRC_HEADERS})
endif()
install (TARGETS blenlib DESTINATION lib)
