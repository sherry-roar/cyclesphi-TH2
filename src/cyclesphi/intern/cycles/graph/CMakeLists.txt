# SPDX-License-Identifier: Apache-2.0
# Copyright 2011-2022 Blender Foundation

set(INC
  ..
)

set(SRC
  node.cpp
  node_type.cpp
  node_xml.cpp
)

set(SRC_HEADERS
  node.h
  node_enum.h
  node_type.h
  node_xml.h
)

set(LIB
  cycles_util
)

if(WITH_CLIENT_CACHE)
	add_definitions(-DWITH_CLIENT_CACHE)
endif()

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_graph "${LIB}" ${SRC} ${SRC_HEADERS})
