# SPDX-License-Identifier: Apache-2.0
# Copyright 2011-2022 Blender Foundation

set(INC
  ..
  ../../glew-mx
  ../../guardedalloc
  ../../mikktspace
  ../../../source/blender/makesdna
  ../../../source/blender/makesrna
  ../../../source/blender/blenlib
  ../../../source/blender/gpu
  ../../../client/api
  ${CMAKE_BINARY_DIR}/source/blender/makesrna/intern
)

set(INC_SYS
  ${PYTHON_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIR}
)

set(SRC
  camera.cpp
  device.cpp
  display_driver.cpp
  image.cpp
  geometry.cpp
  light.cpp
  mesh.cpp
  object.cpp
  object_cull.cpp
  output_driver.cpp
  particles.cpp
  pointcloud.cpp
  curves.cpp
  logging.cpp
  python.cpp
  session.cpp
  shader.cpp
  sync.cpp
  texture.cpp
  viewport.cpp
  volume.cpp

  CCL_api.h
  device.h
  display_driver.h
  id_map.h
  image.h
  object_cull.h
  output_driver.h
  sync.h
  session.h
  texture.h
  util.h
  viewport.h
)

set(LIB
  cycles_bvh
  cycles_device
  cycles_graph
  cycles_kernel
  cycles_scene
  cycles_session
  cycles_subd
  cycles_util

  ${PYTHON_LINKFLAGS}
  ${PYTHON_LIBRARIES}
)

if(WITH_CYCLES_LOGGING)
  list(APPEND LIB
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
  )
endif()

set(ADDON_FILES
  addon/__init__.py
  addon/camera.py
  addon/engine.py
  addon/operators.py
  addon/osl.py
  addon/presets.py
  addon/properties.py
  addon/ui.py
  addon/version_update.py
)

add_definitions(${GL_DEFINITIONS})

if(WITH_CYCLES_DEVICE_HIP)
  add_definitions(-DWITH_HIP)
endif()

if(WITH_CYCLES_DEVICE_METAL)
  add_definitions(-DWITH_METAL)
endif()

if(WITH_MOD_FLUID)
  add_definitions(-DWITH_FLUID)
endif()

if(WITH_OPENVDB)
  add_definitions(-DWITH_OPENVDB ${OPENVDB_DEFINITIONS})
  list(APPEND INC_SYS
    ${OPENVDB_INCLUDE_DIRS}
  )
  list(APPEND LIB
    ${OPENVDB_LIBRARIES}
  )
endif()

if(WITH_ALEMBIC)
  add_definitions(-DWITH_ALEMBIC)
  list(APPEND INC_SYS
    ${ALEMBIC_INCLUDE_DIRS}
  )
  list(APPEND LIB
    ${ALEMBIC_LIBRARIES}
  )
endif()

if(WITH_CLIENT_RAAS)
  add_definitions(-DWITH_CLIENT_RAAS)
endif()

if(WITH_CLIENT_MPI)
  add_definitions(-DWITH_CLIENT_MPI)
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

if(WITH_CLIENT_MPI_CACHE)
	add_definitions(-DWITH_CLIENT_MPI_CACHE)
endif()

if(WITH_CLIENT_RENDERENGINE_SENDER)
 	add_definitions(-DWITH_CLIENT_RENDERENGINE_SENDER)
endif()

if(WITH_OPENIMAGEDENOISE)
  add_definitions(-DWITH_OPENIMAGEDENOISE)
  list(APPEND INC_SYS
    ${OPENIMAGEDENOISE_INCLUDE_DIRS}
  )
endif()

blender_add_lib(bf_intern_cycles "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")

add_dependencies(bf_intern_cycles bf_rna)

delayed_install(${CMAKE_CURRENT_SOURCE_DIR} "${ADDON_FILES}" ${CYCLES_INSTALL_PATH})
