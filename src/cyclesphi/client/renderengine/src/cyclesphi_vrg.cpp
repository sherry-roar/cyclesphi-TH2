#include "cyclesphi.h"

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB

#ifdef _WIN32
#  define VRG_STREAM_API extern "C" __declspec(dllexport)
#else
#  define VRG_STREAM_API
#endif

#  include "vrgstream.h"

#endif

#include <iostream>
#include <omp.h>
#include <string.h>
#include <string>

#include <stdlib.h>

#ifdef WITH_CLIENT_RENDERENGINE_ULTRAGRID_LIB
VRG_STREAM_API enum VrgStreamApiError vrgStreamInit(enum VrgInputFormat inputFormat)
{
  re_init();
  return Ok;
}

VRG_STREAM_API enum VrgStreamApiError vrgStreamRenderFrame(struct RenderPacket *packet)
{
  re_render_frame((char *)packet);
  return Ok;
}

VRG_STREAM_API enum VrgStreamApiError vrgStreamSubmitFrame(struct RenderPacket *packet,
                                                           void *sbs_image_data,
                                                           enum VrgMemory api)
{
  re_submit_frame((char *)packet, (char *)sbs_image_data);
  return Ok;
}

#endif
