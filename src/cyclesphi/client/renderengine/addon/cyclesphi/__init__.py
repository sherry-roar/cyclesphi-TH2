# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "CyclesPhi",
    "author" : "Milan Jaros, Petr Strakos",
    "description" : "",
    "blender" : (3, 3, 0),
    "version" : (0, 0, 4),
    "location" : "",
    "warning" : "",
    "category" : "Render"
}

import bpy
#import bgl
import time

import sys
import ctypes
import numpy
import textwrap
import weakref
import threading

import platform
import numpy as np
import bgl

from gpu_extras.presets import draw_texture_2d

import math

import ctypes
import sys
from ctypes import cdll

from dataclasses import dataclass

from ctypes import Array, cdll, c_void_p, c_char, c_char_p, c_int, c_int32, c_uint32, c_float, c_bool, c_ulong, POINTER

if sys.platform == 'linux':
    gl = cdll.LoadLibrary('libGL.so')
elif sys.platform == 'darwin':
    # ToDo: fix this reference
    gl = cdll.LoadLibrary('/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib')
else:
    gl = ctypes.windll.opengl32

gl.glViewport.argtypes = [c_int32, c_int32, c_uint32, c_uint32]    

# platform specific library loading

try:
    if sys.platform == 'linux':
        _cyclesphi_dll = cdll.LoadLibrary('libcyclesphi_renderengine.so')
    elif sys.platform == 'darwin':
        # ToDo: fix this reference
        _cyclesphi_dll = cdll.LoadLibrary('libcyclesphi_renderengine.dylib')
    else:
        _cyclesphi_dll = cdll.LoadLibrary('cyclesphi_renderengine.dll')


    render_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)    

    _cyclesphi_dll.register_render_callback.argtypes = [c_void_p]

    # CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD resize(unsigned width, unsigned height)
    _cyclesphi_dll.resize.argtypes = [c_int32, c_int32]   

    _cyclesphi_dll.cuda_gl_map_buffer.argtypes = [c_uint32] 
    _cyclesphi_dll.cuda_gl_unmap_buffer.argtypes = [c_uint32]

    # CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD main_loop();
    #_cyclesphi_dll.main_loop.restype = c_int32   
        # CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD recv_pixels_data();
        # CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD send_cam_data();
    _cyclesphi_dll.recv_pixels_data.restype = c_int32
    _cyclesphi_dll.send_cam_data.restype = c_int32

    # CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD client_init(const char *server, int port_cam, int port_data, int w, int h, int step_samples);
    _cyclesphi_dll.client_init.argtypes = [c_char_p, c_int32, c_int32, c_int32, c_int32, c_int32, c_float]

    _cyclesphi_dll.get_DWIDTH.restype = c_int32

    # CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD client_close_connection();

    # CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD get_camera(float *cameratoworld);
    #_cyclesphi_dll.get_camera.argtypes = [c_void_p]

    # set_camera(void *view_martix,
    #                     float lens,
    #                     float nearclip,
    #                     float farclip,
    #                     float sensor_width,
    #                     float sensor_height,
    #                     int sensor_fit,
    #                     float view_camera_zoom,
    #                     float view_camera_offset0,
    #                     float view_camera_offset1,
    #                     int use_view_camera);
    _cyclesphi_dll.set_camera.argtypes = [c_void_p, c_float, c_float, c_float, c_float, c_float, c_int, c_float, c_float, c_float, c_int, c_float, c_float]
    _cyclesphi_dll.set_camera_right.argtypes = [c_void_p, c_float, c_float, c_float, c_float, c_float, c_int, c_float, c_float, c_float, c_int, c_float, c_float]

    _cyclesphi_dll.set_camera_ug.argtypes = [c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float]

    _cyclesphi_dll.set_camera_right_ug.argtypes = [c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float,c_float]

    # CYCLESPHI_EXPORT_DLL int CYCLESPHI_EXPORT_STD get_samples();
    _cyclesphi_dll.get_samples.restype = c_int32   
    _cyclesphi_dll.get_current_samples.restype = c_int32   

    # CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD set_samples(int s);
    #_cyclesphi_dll.set_samples.argtypes = [c_int32]

    # CYCLESPHI_EXPORT_DLL void CYCLESPHI_EXPORT_STD get_pixels(char* pixels)
    _cyclesphi_dll.get_pixels.argtypes = [c_void_p]
    _cyclesphi_dll.get_pixels_right.argtypes = [c_void_p]

    #_cyclesphi_dll.get_renderengine_type.restype = c_int32
except:
    pass     
#-------------------------------------------------------------------

import bpy
import re
import os
import math
import blf
import sys
from bpy.props import IntProperty, CollectionProperty, BoolProperty, StringProperty, FloatProperty
from bpy.types import Panel, UIList, Operator 
from mathutils import Vector, Matrix
from math import pi, sqrt, sin, cos, pow, atan, tan
#from bgl import *

# def get_user_settings():
#     return bpy.context.preferences.addons['cyclesphi'].preferences.settings

def check_gl_error():
    error = bgl.glGetError()
    if error != bgl.GL_NO_ERROR:
        raise Exception(error)
#############################################

class CyclesPhiServerSettings(bpy.types.PropertyGroup):
    port_cam: bpy.props.IntProperty(
        name="Port Cam",
        min=0, 
        max=65565, 
        default=6004
    )    

    port_data: bpy.props.IntProperty(
        name="Port Data",
        min=0, 
        max=65565, 
        default=6005
    )  

    server_name: bpy.props.StringProperty(
        name="Server",
        default="localhost"
    )       

    vrclient_path: bpy.props.StringProperty(
        name="Client",
        default="c:/Users/jar091/projects/work/blender_client_build/vrclient/Debug/vrclient.exe",
        subtype="FILE_PATH"
    )

    ssh_command: bpy.props.StringProperty(
        name="ssh",
        default="-i c:/Users/jar091/projects/cert/local/ans.ssh milanjaros@195.113.175.2 ./run_vr.sh mercedes"
    )        

    monitor: bpy.props.IntProperty(
        name="Monitor",
        min=-1, 
        max=3, 
        default=1
    )  

class CyclesPhiSamplingSettings(bpy.types.PropertyGroup):
    step_samples: bpy.props.IntProperty(
        name="Step Samples",
        min=1, 
        max=65565, 
        default=1
    )   

    width: bpy.props.IntProperty(
        name="Width",
        min=16, 
        max=65565, 
        default=1920
    )    

    height: bpy.props.IntProperty(
        name="Height",
        min=16, 
        max=65565, 
        default=1080
    )

    right_eye: bpy.props.FloatProperty(
        name="Right Eye",
        min=0.0, 
        max=100.0, 
        default=0.035
    )

    use_viewport: bpy.props.BoolProperty(
        name="Use Viewport",
        default=True
    )                    

class CyclesPhiRenderSettings(bpy.types.PropertyGroup):
    server_settings: bpy.props.PointerProperty(type=CyclesPhiServerSettings)
    sampling_settings: bpy.props.PointerProperty(type=CyclesPhiSamplingSettings)
    #client_process_id: bpy.props.IntProperty(default=-1)

class CyclesPhiData:
    def __init__(self):
        self.client_process_id = None
        self.ssh_process_id = None
        self.cyclesphi_context = None


class GLTexture:
    channels = 4

    def __init__(self, width, height):
        self.width = width
        self.height = height

        textures = bgl.Buffer(bgl.GL_INT, [1,])
        bgl.glGenTextures(1, textures)
        self.texture_id = textures[0]

        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture_id)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_REPEAT)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_REPEAT)

        bgl.glTexImage2D(
            bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA,
            self.width, self.height, 0,
            bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE,
            bgl.Buffer(bgl.GL_BYTE, [self.width, self.height, self.channels])
        )

    def __del__(self):
        textures = bgl.Buffer(bgl.GL_INT, [1, ], [self.texture_id, ])
        bgl.glDeleteTextures(1, textures)

    def set_image(self, im: np.array):
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(
            bgl.GL_TEXTURE_2D, 0,
            0, 0, self.width, self.height,
            bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(im.ctypes.data)
        )
class CyclesPhiContext:
    channels = 4
    send_cam_data_result = -1

    def __init__(self):
        self.server = None
        self.port_cam = None
        self.port_data = None
        self.width = None
        self.height = None        
        self.step_samples = None      
        self.righ_eye = None

        self.data = None  
        #self.data_right = None

    def init(self, context, server, port_cam, port_data, width, height, step_samples, righ_eye):
  
        self.server = server
        self.port_cam = port_cam
        self.port_data = port_data
        self.width = width
        self.height = height       
        self.step_samples = step_samples     
        self.righ_eye = righ_eye

        self.data = np.empty((height, width, self.channels), dtype=np.uint8)
        #self.data_right = np.empty((height, width, self.channels), dtype=np.uint8)

        print(self.server.encode(), self.port_cam, self.port_data, self.width, self.height, self.step_samples, self.righ_eye)

    def client_init(self):
        _cyclesphi_dll.client_init(self.server.encode(), self.port_cam, self.port_data, self.width, self.height, self.step_samples, self.righ_eye)

        self.g_width = self.width
        self.g_height = self.height
        self.DWIDTH = _cyclesphi_dll.get_DWIDTH()

        #GLuint g_bufferIds[3];   # IDs of PBO
        #self.g_bufferIds = bgl.Buffer(bgl.GL_INT, 3)
        #GLuint g_textureIds[3];  # ID of texture        
        #self.g_textureIds = bgl.Buffer(bgl.GL_INT, 3)

        #self.setupTexture(2)

        check_gl_error()

    def client_close_connection(self):
        _cyclesphi_dll.client_close_connection()

    # def resize(self, width, height):
    #     #self.init(self.server, self.port_cam, self.port_data, width, height, self.step_samples)
    #     pass 
    # 

    def register_render_callback(self, rc):
        _cyclesphi_dll.register_render_callback(rc)   

    def render(self, restart=False, tile=None):
        _cyclesphi_dll.send_cam_data()
        _cyclesphi_dll.recv_pixels_data()

        return 1

    def set_camera(self, camera_data):
        transformL = np.array(camera_data.transform, dtype=np.float32)
        #transformR = np.array(camera_dataR.transform, dtype=np.float32)

        _cyclesphi_dll.set_camera(transformL.ctypes.data, 
                                        camera_data.focal_length,
                                        camera_data.clip_plane[0],
                                        camera_data.clip_plane[1],
                                        camera_data.sensor_size[0],
                                        camera_data.sensor_size[1],
                                        camera_data.sensor_fit,
                                        camera_data.view_camera_zoom,
                                        camera_data.view_camera_offset[0],
                                        camera_data.view_camera_offset[1],
                                        camera_data.use_view_camera,
                                        camera_data.shift_x,
                                        camera_data.shift_y)

        # _cyclesphi_dll.set_camera_right(transformR.ctypes.data, 
        #                                 camera_dataR.focal_length,
        #                                 camera_dataR.clip_plane[0],
        #                                 camera_dataR.clip_plane[1],
        #                                 camera_dataR.sensor_size[0],
        #                                 camera_dataR.sensor_size[1],
        #                                 camera_dataR.sensor_fit,
        #                                 camera_dataR.view_camera_zoom,
        #                                 camera_dataR.view_camera_offset[0],
        #                                 camera_dataR.view_camera_offset[1],
        #                                 camera_dataR.use_view_camera,
        #                                 camera_dataR.shift_x,
        #                                 camera_dataR.shift_y)

        ss = camera_data.sensor_size[0]
        if camera_data.use_view_camera == 0:
            ss = -ss

        _cyclesphi_dll.set_camera_ug( 
                                        float(camera_data.focal_length),
                                        float(ss),
                                        float(camera_data.view_camera_zoom),
                                        float(camera_data.shift_x),
                                        float(camera_data.pos[0]),
                                        float(camera_data.pos[1]),
                                        float(camera_data.pos[2]),
                                        float(camera_data.quat[1]),
                                        float(camera_data.quat[2]),
                                        float(camera_data.quat[3]),
                                        float(camera_data.quat[0])
                                        )

        # ss = camera_dataR.sensor_size[0]
        # if camera_dataR.use_view_camera == 0:
        #     ss = -ss

        # _cyclesphi_dll.set_camera_right_ug( 
        #                                 float(camera_dataR.focal_length),
        #                                 float(ss),
        #                                 float(camera_dataR.view_camera_zoom),
        #                                 float(camera_dataR.shift_x),
        #                                 float(camera_dataR.pos[0]),
        #                                 float(camera_dataR.pos[1]),
        #                                 float(camera_dataR.pos[2]),
        #                                 float(camera_dataR.quat[1]),
        #                                 float(camera_dataR.quat[2]),
        #                                 float(camera_dataR.quat[3]),
        #                                 float(camera_dataR.quat[0])
        #                                 )                                        

        # cam_thread = threading.Thread(target=self.send_cam_data)
        # cam_thread.start()        

    def get_image(self):
        _cyclesphi_dll.get_pixels(ctypes.c_void_p(self.data.ctypes.data))
        return self.data

    # def get_image_right(self):
    #     _cyclesphi_dll.get_pixels_right(ctypes.c_void_p(self.data_right.ctypes.data))
    #     return self.data_right        

    def get_current_samples(self):
        return _cyclesphi_dll.get_current_samples()

    # def toOrtho(self, eye, width, height):        
    #     # set viewport to be the entire window
    #     gl.glViewport(0, 0, int(width), int(height))

    #     # return

    #     # # set orthographic viewing frustum
    #     # gl.glMatrixMode(0x1701) #gl.GL_PROJECTION)
    #     # gl.glLoadIdentity()

    #     # if eye == 2:
    #     #     gl.glOrtho(0, 1, 0, 1, -1, 1)

    #     # if eye == 0:
    #     #     gl.glOrtho(0, 0.5, 0, 1, -1, 1)

    #     # if eye == 1:
    #     #     gl.glOrtho(0.5, 1, 0, 1, -1, 1)

    #     # # switch to modelview matrix in order to set scene
    #     # gl.glMatrixMode(0x1700) #bgl.GL_MODELVIEW)
    #     # gl.glLoadIdentity()

    # def freeTexture(self, eye):
    #     bgl.glDeleteTextures(1, self.g_textureIds[eye])
    #     if eye == 0 or eye == 1:
    #         bgl.glDeleteBuffers(1, self.g_bufferIds[eye])

    #     if eye == 2:
    #         #cuda_assert(cudaGLUnmapBufferObject(g_bufferIds[eye]))
    #         #cuda_assert(cudaGLUnregisterBufferObject(g_bufferIds[eye]))
    #         _cyclesphi_dll.cuda_gl_umap_buffer(self.g_bufferIds[eye])

    #         bgl.glDeleteFramebuffers(1, self.g_bufferIds[eye])
        

    # # Setup Texture
    # def setupTexture(self, eye):
    #     #if 1
    #     #bgl.GLuint pboIds[1]      # IDs of PBO
    #     pboIds = bgl.Buffer(bgl.GL_INT, [1,])
    #     #bgl.GLuint textureIds[1]  # ID of texture
    #     textureIds = bgl.Buffer(bgl.GL_INT, [1,])

    #     # init 2 texture objects
    #     bgl.glGenTextures(1, textureIds)
    #     self.g_textureIds[eye] = textureIds[0]

    #     bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.g_textureIds[eye])
    #     bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_NEAREST)
    #     bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_NEAREST)
    #     bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_REPEAT)
    #     bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_REPEAT)
        
    #     #ifdef WITH_CLIENT_YUV
    #     #glTexImage2D(
    #     #    GL_TEXTURE_2D, 0, GL_LUMINANCE8, DWIDTH, g_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL)
    #     #else
    #     if eye == 2:
    #         dw = self.DWIDTH
    #     else: 
    #         dw = self.DWIDTH / 2

    #     bgl.glTexImage2D(bgl.GL_TEXTURE_2D,
    #                 0,
    #                 bgl.GL_RGBA8,
    #                 dw,
    #                 self.g_height,
    #                 0,
    #                 bgl.GL_RGBA,
    #                 bgl.GL_UNSIGNED_BYTE,
    #                 0)

    #     bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

    #     # if eye == 0 or eye == 1:
    #     #     # create 2 pixel buffer objects, you need to delete them when program exits.
    #     #     # glBufferData() with NULL pointer reserves only memory space.
    #     #     bgl.glGenFramebuffers(1, pboIds)
    #     #     self.g_bufferIds[eye] = pboIds[0]

    #     #     bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, self.g_bufferIds[eye])

    #     #     # Set "renderedTexture" as our colour attachement #0
    #     #     #glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g_textureIds[eye], 0)
    #     #     bgl.glFramebufferTexture2D(bgl.GL_FRAMEBUFFER, bgl.GL_COLOR_ATTACHMENT0, bgl.GL_TEXTURE_2D, self.g_textureIds[eye], 0)

    #     #     bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, 0)
        
    #     #endif

    #     # if (eye == 2)       
    #     # create 2 pixel buffer objects, you need to delete them when program exits.
    #     # glBufferData() with NULL pointer reserves only memory space.
    #     bgl.glGenBuffers(1, pboIds)
    #     self.g_bufferIds[eye] = pboIds[0]

    #     bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, self.g_bufferIds[eye])

    #     bgl.glBufferData(bgl.GL_PIXEL_UNPACK_BUFFER,
    #                 dw * self.g_height * 4,
    #                 0,
    #                 bgl.GL_DYNAMIC_COPY)

    #     #cuda_assert(cudaGLRegisterBufferObject(g_bufferIds[eye]))
    #     #cuda_assert(cudaGLMapBufferObject((void **)&g_pixels_buf_d, g_bufferIds[eye]))
    #     _cyclesphi_dll.cuda_gl_map_buffer(self.g_bufferIds[eye])

    #     bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, 0)
        
    @staticmethod
    def _draw_texture(texture_id, x, y, width, height):
        # INITIALIZATION

        # Getting shader program
        shader_program = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

        # Generate vertex array
        vertex_array = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenVertexArrays(1, vertex_array)

        texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
        position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

        # Generate geometry buffers for drawing textured quad
        position = [x, y, x + width, y, x + width, y + height, x, y + height]
        position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
        texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

        vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)
        bgl.glGenBuffers(2, vertex_buffer)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[0])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[1])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)

        # DRAWING
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, texture_id)

        bgl.glBindVertexArray(vertex_array[0])
        bgl.glEnableVertexAttribArray(texturecoord_location)
        bgl.glEnableVertexAttribArray(position_location)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[0])
        bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[1])
        bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)

        bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)

        bgl.glBindVertexArray(0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

        # DELETING
        bgl.glDeleteBuffers(2, vertex_buffer)
        bgl.glDeleteVertexArrays(1, vertex_array)        

    # def gl_render(self, eye):
    #     #if 1
    #     # render to texture
    #     if eye == 0 or eye == 1:
    #         # Left Eye
    #         bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, self.g_bufferIds[eye])
    #         # glViewport(0, 0, g_width / 2, g_height)
    #         #  vr_RenderScene(vr::Eye_Left)
    #         self.toOrtho(eye, self.DWIDTH / 2, self.g_height)
    #     else:

    #         bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.g_textureIds[2])
    #         bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, self.g_bufferIds[2])

    #         # copy pixels from PBO to texture object
    #         # Use offset instead of ponter.

    #         #  ifdef WITH_CLIENT_YUV
    #         # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DWIDTH, g_height, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
    #         #  else
    #         bgl.glTexSubImage2D(bgl.GL_TEXTURE_2D,
    #                         0,
    #                         0,
    #                         0,
    #                         self.DWIDTH,  
    #                         self.g_height,
    #                         bgl.GL_RGBA,
    #                         bgl.GL_UNSIGNED_BYTE,
    #                         0)

    #         bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, 0)
    #         bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

    #         border = (0, 0), (self.DWIDTH, self.height)
    #         #draw_texture_2d(self.g_textureIds[2], border[0], *border[1])
    #         self._draw_texture(self.g_textureIds[2], 0, 0, self.DWIDTH, self.height)
    #         return                        

    #         self.toOrtho(eye, self.g_width, self.g_height)

    #         check_gl_error()

    #         bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.g_textureIds[2])

    #         check_gl_error()

    #         gl.glBegin(7) #bgl.GL_QUADS)

    #         check_gl_error()

    #         gl.glTexCoord2d(0, 0)
    #         gl.glVertex2d(0, 0)
    #         gl.glTexCoord2d(1, 0)
    #         gl.glVertex2d(1, 0)
    #         gl.glTexCoord2d(1, 1)
    #         gl.glVertex2d(1, 1)
    #         gl.glTexCoord2d(0, 1)
    #         gl.glVertex2d(0, 1)

    #         check_gl_error()

    #         gl.glEnd()

    #         check_gl_error()

    #         bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

    #         check_gl_error()

    #         return
        

    #     # glfwMakeContextCurrent(g_windows[eye]);

    #     # bind the texture and PBO
    #     bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.g_textureIds[2])
    #     bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, self.g_bufferIds[2])

    #     # copy pixels from PBO to texture object
    #     # Use offset instead of ponter.

    #     #  ifdef WITH_CLIENT_YUV
    #     # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DWIDTH, g_height, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
    #     #  else
    #     bgl.glTexSubImage2D(bgl.GL_TEXTURE_2D,
    #                     0,
    #                     0,
    #                     0,
    #                     self.DWIDTH,  
    #                     self.g_height,
    #                     bgl.GL_RGBA,
    #                     bgl.GL_UNSIGNED_BYTE,
    #                     0)
    #     #  endif



    #     # draw a point with texture
    #     bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.g_textureIds[2])
    
    #     gl.glBegin(0x0007) #bgl.GL_QUADS)

    #     gl.glTexCoord2d(0, 0)
    #     gl.glVertex2d(0, 0)
    #     gl.glTexCoord2d(1, 0)
    #     gl.glVertex2d(1, 0)
    #     gl.glTexCoord2d(1, 1)
    #     gl.glVertex2d(1, 1)
    #     gl.glTexCoord2d(0, 1)
    #     gl.glVertex2d(0, 1)


    #     gl.glEnd()

    #     # unbind texture
    #     bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, 0)
    #     bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

    #     # render to texture
    #     if eye == 0 or eye == 1:
    #         bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, 0)
                

MAX_ORTHO_DEPTH = 200.0

@dataclass(init=False, eq=True)
class CameraData:
    """ Comparable dataclass which holds all camera settings """

    transform: tuple = None #void *camera_object,
    focal_length: float = None #float lens,
    clip_plane: (float, float) = None #float nearclip, float farclip,
    sensor_size: (float, float) = None #float sensor_width, float sensor_height,
    sensor_fit: int = None # int sensor_fit,
    view_camera_zoom: float = None # float view_camera_zoom,
    view_camera_offset: (float, float) = None # float view_camera_offset0, float view_camera_offset1,
    use_view_camera: int = None # int use_view_camera
    shift_x: float = None
    shift_y: float = None
    quat: tuple = None
    pos: tuple = None

    @staticmethod
    def init_from_camera(camera: bpy.types.Camera, transform, ratio, border=((0, 0), (1, 1))):
        """ Returns CameraData from bpy.types.Camera """

        #pos, size = border

        data = CameraData()
        data.clip_plane = (camera.clip_start, camera.clip_end)
        data.transform = tuple(transform)
        data.use_view_camera = 1

        data.shift_x = camera.shift_x
        data.shift_y = camera.shift_y

        data.quat = tuple(transform.to_quaternion()) 
        data.pos = tuple(transform.to_translation()) 
        
        # projection_matrix = camera.calc_matrix_camera(
        #     bpy.context.view_layer.depsgraph,
        #     x = bpy.context.scene.render.resolution_x,
        #     y = bpy.context.scene.render.resolution_y,
        #     scale_x = bpy.context.scene.render.pixel_aspect_x,
        #     scale_y = bpy.context.scene.render.pixel_aspect_y,
        # )  
        
        # if camera.dof.use_dof:
        #     # calculating focus_distance
        #     if not camera.dof.focus_object:
        #         focus_distance = camera.dof.focus_distance
        #     else:
        #         obj_pos = camera.dof.focus_object.matrix_world.to_translation()
        #         camera_pos = transform.to_translation()
        #         focus_distance = (obj_pos - camera_pos).length

        #     data.dof_data = (max(focus_distance, 0.001),
        #                      camera.dof.aperture_fstop,
        #                      max(camera.dof.aperture_blades, 4))
        # else:
        #     data.dof_data = None

        if camera.sensor_fit == 'VERTICAL':
           # data.lens_shift = (camera.shift_x / ratio, camera.shift_y)
           data.sensor_fit = 2
        elif camera.sensor_fit == 'HORIZONTAL':
            #data.lens_shift = (camera.shift_x, camera.shift_y * ratio)
            data.sensor_fit = 1
        elif camera.sensor_fit == 'AUTO':
            # data.lens_shift = (camera.shift_x, camera.shift_y * ratio) if ratio > 1.0 else \
            #     (camera.shift_x / ratio, camera.shift_y)
            data.sensor_fit = 0
        else:
            raise ValueError("Incorrect camera.sensor_fit value", camera, camera.sensor_fit)

        # data.lens_shift = tuple(data.lens_shift[i] / size[i] + (pos[i] + size[i] * 0.5 - 0.5) / size[i] for i in (0, 1))

        if camera.type == 'PERSP':
            #data.mode = pyrpr.CAMERA_MODE_PERSPECTIVE
            data.focal_length = camera.lens
            #data.sensor_fit = camera.sensor_fit
            data.sensor_size = (camera.sensor_width, camera.sensor_height)
            # if camera.sensor_fit == 'VERTICAL':
            #     data.sensor_size = (camera.sensor_height * ratio, camera.sensor_height)
            # elif camera.sensor_fit == 'HORIZONTAL':
            #     data.sensor_size = (camera.sensor_width, camera.sensor_width / ratio)
            # else:
            #     data.sensor_size = (camera.sensor_width, camera.sensor_width / ratio) if ratio > 1.0 else \
            #                        (camera.sensor_width * ratio, camera.sensor_width)

            # data.sensor_size = tuple(data.sensor_size[i] * size[i] for i in (0, 1))

            #data.fov = 2.0 * math.atan(0.5 * camera.sensor_width / camera.lens / ratio )


        # elif camera.type == 'ORTHO':
        #     #data.mode = pyrpr.CAMERA_MODE_ORTHOGRAPHIC
        #     if camera.sensor_fit == 'VERTICAL':
        #         data.ortho_size = (camera.ortho_scale * ratio, camera.ortho_scale)
        #     elif camera.sensor_fit == 'HORIZONTAL':
        #         data.ortho_size = (camera.ortho_scale, camera.ortho_scale / ratio)
        #     else:
        #         data.ortho_size = (camera.ortho_scale, camera.ortho_scale / ratio) if ratio > 1.0 else \
        #                           (camera.ortho_scale * ratio, camera.ortho_scale)

        #     data.ortho_size = tuple(data.ortho_size[i] * size[i] for i in (0, 1))
        #     data.clip_plane = (camera.clip_start, min(camera.clip_end, MAX_ORTHO_DEPTH + camera.clip_start))

        # elif camera.type == 'PANO':
        #     # TODO: Recheck parameters for PANO camera
        #     #data.mode = pyrpr.CAMERA_MODE_LATITUDE_LONGITUDE_360
        #     data.focal_length = camera.lens
        #     if camera.sensor_fit == 'VERTICAL':
        #         data.sensor_size = (camera.sensor_height * ratio, camera.sensor_height)
        #     elif camera.sensor_fit == 'HORIZONTAL':
        #         data.sensor_size = (camera.sensor_width, camera.sensor_width / ratio)
        #     else:
        #         data.sensor_size = (camera.sensor_width, camera.sensor_width / ratio) if ratio > 1.0 else \
        #                            (camera.sensor_width * ratio, camera.sensor_width)

        #     data.sensor_size = tuple(data.sensor_size[i] * size[i] for i in (0, 1))

        else:
            raise ValueError("Incorrect camera.type value",camera, camera.type)

        return data

    @staticmethod
    def init_from_context(context: bpy.types.Context):
        """ Returns CameraData from bpy.types.Context """

        VIEWPORT_SENSOR_SIZE = 72.0     # this constant was found experimentally, didn't find such option in
                                        # context.space_data or context.region_data

        ratio = context.region.width / context.region.height
        if context.region_data.view_perspective == 'PERSP':
            data = CameraData()
            #data.mode = pyrpr.CAMERA_MODE_PERSPECTIVE
            data.clip_plane = (context.space_data.clip_start, context.space_data.clip_end)
            #data.lens_shift = (0.0, 0.0)
            # data.sensor_size = (VIEWPORT_SENSOR_SIZE, VIEWPORT_SENSOR_SIZE / ratio) if ratio > 1.0 else \
            #                    (VIEWPORT_SENSOR_SIZE * ratio, VIEWPORT_SENSOR_SIZE)
            #data.fov = 2.0 * math.atan(0.5 * VIEWPORT_SENSOR_SIZE / context.space_data.lens / ratio )
            vmat = context.region_data.view_matrix.inverted()
            data.transform = tuple(vmat)
            data.focal_length = context.space_data.lens
            data.use_view_camera = 0
            data.sensor_size = (VIEWPORT_SENSOR_SIZE, VIEWPORT_SENSOR_SIZE)
            data.view_camera_offset = (0,0)
            data.view_camera_zoom = 1.0
            data.sensor_fit=0
            data.shift_x = 0
            data.shift_y = 0

            data.quat = tuple(vmat.to_quaternion()) 
            data.pos = tuple(vmat.to_translation())      


        # elif context.region_data.view_perspective == 'ORTHO':
        #     data = CameraData()
        #     #data.mode = pyrpr.CAMERA_MODE_ORTHOGRAPHIC
        #     #ortho_size = context.region_data.view_distance * VIEWPORT_SENSOR_SIZE / context.space_data.lens
        #     #data.lens_shift = (0.0, 0.0)
        #     ortho_depth = min(context.space_data.clip_end, MAX_ORTHO_DEPTH)
        #     data.clip_plane = (-ortho_depth * 0.5, ortho_depth * 0.5)
        #     #data.ortho_size = (ortho_size, ortho_size / ratio) if ratio > 1.0 else \
        #     #                  (ortho_size * ratio, ortho_size)

        #     data.transform = tuple(context.region_data.view_matrix.inverted())

        elif context.region_data.view_perspective == 'CAMERA':
            camera_obj = context.space_data.camera
            data = CameraData.init_from_camera(camera_obj.data, context.region_data.view_matrix.inverted(), ratio)
            # data = CameraData()
            # data.clip_plane = (camera_obj.data.clip_start, camera_obj.data.clip_end)
            # data.transform = tuple(context.region_data.view_matrix.inverted())
            # data.fov = 2.0 * atan(0.5 * camera_obj.data.sensor_width / camera_obj.data.lens / ratio )
            #data.transform = tuple(camera_obj.matrix_world) #tuple(context.region_data.view_matrix.inverted())               

            # # This formula was taken from previous plugin with corresponded comment
            # # See blender/intern/cycles/blender/blender_camera.cpp:blender_camera_from_view (look for 1.41421f)
            # zoom = 4.0 / (2.0 ** 0.5 + context.region_data.view_camera_zoom / 50.0) ** 2
            data.view_camera_zoom = context.region_data.view_camera_zoom

            # # Updating lens_shift due to viewport zoom and view_camera_offset
            # # view_camera_offset should be multiplied by 2
            # data.lens_shift = ((data.lens_shift[0] + context.region_data.view_camera_offset[0] * 2) / zoom,
            #                    (data.lens_shift[1] + context.region_data.view_camera_offset[1] * 2) / zoom)
            data.view_camera_offset = (context.region_data.view_camera_offset[0], context.region_data.view_camera_offset[1])
            # if data.mode == pyrpr.CAMERA_MODE_ORTHOGRAPHIC:
            #     data.ortho_size = (data.ortho_size[0] * zoom, data.ortho_size[1] * zoom)
            # else:
            #     data.sensor_size = (data.sensor_size[0] * zoom, data.sensor_size[1] * zoom)

        else:
            raise ValueError("Incorrect view_perspective value", context.region_data.view_perspective)

        return data         

@dataclass(init=False, eq=True)
class ViewportSettings:
    """
    Comparable dataclass which holds render settings for ViewportEngine:
    - camera viewport settings
    - render resolution
    - screen resolution
    - render border
    """

    camera_data: CameraData
    #camera_dataR: CameraData
    width: int
    height: int
    screen_width: int
    screen_height: int
    border: tuple

    def __init__(self, context: bpy.types.Context):
        """Initializes settings from Blender's context"""
        # if _cyclesphi_dll.get_renderengine_type() == 1:
        #     self.camera_data,self.camera_dataR = CameraData.init_from_context_openvr(context)
        #     self.screen_height = context.scene.openvr_user_prop.openVrGlRenderer.getHeight()
        #     self.screen_width = context.scene.openvr_user_prop.openVrGlRenderer.getWidth()   
        # else:
        self.camera_data = CameraData.init_from_context(context)
        self.screen_width, self.screen_height = context.region.width, context.region.height        

        scene = context.scene

        # getting render border
        x1, y1 = 0, 0
        x2, y2 = self.screen_width, self.screen_height  

        # getting render resolution and render border
        self.width, self.height = x2 - x1, y2 - y1
        self.border = (x1, y1), (self.width, self.height)              

class Engine:
    """ This is the basic Engine class """

    def __init__(self, cyclesphi_engine):
        self.cyclesphi_engine = weakref.proxy(cyclesphi_engine)
        self.cyclesphi_context = CyclesPhiContext()
        bpy.context.scene.cyclesphi_data.cyclesphi_context = self.cyclesphi_context

#g_viewport_engine = None

class ViewportEngine(Engine):
    """ Viewport render engine """
    def __init__(self, cyclesphi_engine):
        super().__init__(cyclesphi_engine)

        self.gl_texture: GLTexture = None
        self.viewport_settings: ViewportSettings = None

        self.sync_render_thread: threading.Thread = None
        self.restart_render_event = threading.Event()
        self.render_lock = threading.Lock()
        self.resolve_lock = threading.Lock()

        self.is_finished = True
        self.is_synced = False
        self.is_rendered = False
        #self.is_denoised = False
        self.is_resized = False

        #self.render_iterations = 0
        #self.render_time = 0

        #g_viewport_engine = self
        self.render_callback = render_callback_type(self.render_callback)

    def stop_render(self):
        print("stop_render")
        self.is_finished = True
        
        self.restart_render_event.set()
        self.sync_render_thread.join()

        #self.cyclesphi_context.client_close_connection()
        self.cyclesphi_context = None
        self.image_filter = None
        pass

    def _do_sync_render(self, depsgraph):
        """
        Thread function for self.sync_render_thread. It always run during viewport render.
        If it doesn't render it waits for self.restart_render_event
        """

        def notify_status(info, status):
            """ Display export progress status """
            wrap_info = textwrap.fill(info, 120)
            self.cyclesphi_engine.update_stats(status, wrap_info)
            #log(status, wrap_info)

            # requesting blender to call draw()
            self.cyclesphi_engine.tag_redraw()

        class FinishRender(Exception):
            pass

        # print('Start _do_sync_render')
        # self.cyclesphi_context.client_init()

        try:
            # SYNCING OBJECTS AND INSTANCES
            notify_status("Starting...", "Sync")
            time_begin = time.perf_counter()      

            self.is_synced = True

            # RENDERING
            notify_status("Starting...", "Render")

            # Infinite cycle, which starts when scene has to be re-rendered.
            # It waits for restart_render_event be enabled.
            # Exit from this cycle is implemented through raising FinishRender
            # when self.is_finished be enabled from main thread.
            while True:
                self.restart_render_event.wait()

                if self.is_finished:
                    raise FinishRender

                # preparations to start rendering
                iteration = 0
                time_begin = 0.0
                # if is_adaptive:
                #     all_pixels = active_pixels = self.cyclesphi_context.width * self.cyclesphi_context.height
                #is_last_iteration = False

                # this cycle renders each iteration
                while True:
                    if self.is_finished:
                        raise FinishRender                          

                    if self.restart_render_event.is_set():
                        # clears restart_render_event, prepares to start rendering
                        self.restart_render_event.clear()
                        iteration = 0

                        if self.is_resized:
                            #if not self.cyclesphi_context.gl_interop:
                            # When gl_interop is not enabled, than resize is better to do in
                            # this thread. This is important for hybrid.
                            with self.render_lock:
                                    self.cyclesphi_context.resize(self.viewport_settings.width,
                                                            self.viewport_settings.height)
                            self.is_resized = False

                        time_begin = time.perf_counter()

                    # rendering
                    with self.render_lock:
                        if self.restart_render_event.is_set():
                            break

                    self.cyclesphi_context.render(restart=(iteration == 0))

                    self.is_rendered = True     
                    current_samples = self.cyclesphi_context.get_current_samples()

                    time_render = time.perf_counter() - time_begin
                    fps = current_samples / time_render
                    info_str = f"Time: {time_render:.1f} sec"\
                               f" | Samples: {current_samples}" \
                               f" | FPS: {fps:.1f}"


                    notify_status(info_str, "Render")

        except FinishRender:
            print("Finish by user")
            pass

        except Exception as e:
            print(e)
            self.is_finished = True

            # notifying viewport about error
            notify_status(f"{e}.\nPlease see logs for more details.", "ERROR")


        #self.cyclesphi_context.client_close_connection()

        print('Finish _do_sync_render')

    #@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    def render_callback(self, sample):
        # requesting blender to call draw()
        # if g_viewport_engine != None:
        self.cyclesphi_engine.tag_redraw()  
        #print("render_callback")

        return 0

    # #@ctypes.CFUNCTYPE(None)
    # def render_callback(self, sample):
    #     # requesting blender to call draw()
    #     self.cyclesphi_engine.tag_redraw()                

    def sync(self, context, depsgraph):
        print('Start sync')

        scene = depsgraph.scene
        view_layer = depsgraph.view_layer

        # getting initial render resolution
        if scene.cyclesphi.sampling_settings.use_viewport == True:
            viewport_settings = ViewportSettings(context)
            width, height = viewport_settings.width, viewport_settings.height
            if width * height == 0:
                # if width, height == 0, 0, then we set it to 1, 1 to be able to set AOVs
                width, height = 1, 1

            scene.cyclesphi.sampling_settings.width = width
            scene.cyclesphi.sampling_settings.height = height

        #client_init(const char *server, int port_cam, int port_data, int w, int h, int step_samples)
        self.cyclesphi_context.init(context, scene.cyclesphi.server_settings.server_name, 
                                    scene.cyclesphi.server_settings.port_cam,
                                    scene.cyclesphi.server_settings.port_data,                                    
                                    scene.cyclesphi.sampling_settings.width, 
                                    scene.cyclesphi.sampling_settings.height,
                                    scene.cyclesphi.sampling_settings.step_samples,
                                    scene.cyclesphi.sampling_settings.right_eye)
        # if not self.cyclesphi_context.gl_interop:
        self.gl_texture = GLTexture(width, height)        
        self.cyclesphi_context.client_init()
                 
        self.is_finished = False
        
        # if _cyclesphi_dll.get_renderengine_type() == 2:
        #     self.is_synced = True
        #     self.is_rendered = True
        #     self.cyclesphi_context.register_render_callback(self.render_callback)
        # else:
        print('Start _do_sync_render')
        self.restart_render_event.clear()
        self.sync_render_thread = threading.Thread(target=self._do_sync_render, args=(depsgraph,))
        self.sync_render_thread.start()

        print('Finish sync')        

    @staticmethod
    def _draw_texture(texture_id, x, y, width, height):
        # INITIALIZATION

        # Getting shader program
        shader_program = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

        # Generate vertex array
        vertex_array = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenVertexArrays(1, vertex_array)

        texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
        position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

        # Generate geometry buffers for drawing textured quad
        position = [x, y, x + width, y, x + width, y + height, x, y + height]
        position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
        texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

        vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)
        bgl.glGenBuffers(2, vertex_buffer)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[0])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[1])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)

        # DRAWING
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, texture_id)

        bgl.glBindVertexArray(vertex_array[0])
        bgl.glEnableVertexAttribArray(texturecoord_location)
        bgl.glEnableVertexAttribArray(position_location)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[0])
        bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[1])
        bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)

        bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)

        bgl.glBindVertexArray(0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

        # DELETING
        bgl.glDeleteBuffers(2, vertex_buffer)
        bgl.glDeleteVertexArrays(1, vertex_array)

    def _get_render_image(self):
        ''' This is only called for non-GL interop image gets '''
        # if utils.IS_MAC:
        #     with self.render_lock:
        #         return self.cyclesphi_context.get_image()
        # else:
        return self.cyclesphi_context.get_image() #, self.cyclesphi_context.get_image_right()

    def draw(self, context):
        #log("Draw")

        if not self.is_synced or self.is_finished:
            return

        scene = context.scene

        with self.render_lock:
            viewport_settings = ViewportSettings(context)

            if viewport_settings.width * viewport_settings.height == 0:
                return

            if viewport_settings.camera_data is None: # or viewport_settings.camera_dataR is None:
                return                

            if self.viewport_settings != viewport_settings:
                #viewport_settings.export_camera(self.cyclesphi_context.scene.camera)
                self.cyclesphi_context.set_camera(viewport_settings.camera_data) #, viewport_settings.camera_dataR)                
                self.viewport_settings = viewport_settings

                if self.cyclesphi_context.width != viewport_settings.width \
                        or self.cyclesphi_context.height != viewport_settings.height:

                    resolution = (viewport_settings.width, viewport_settings.height)
                    
                    with self.resolve_lock:
                            self.cyclesphi_context.resize(*resolution)

                    if self.gl_texture:
                        self.gl_texture = GLTexture(*resolution)

                    self.is_resized = True

                #if _cyclesphi_dll.get_renderengine_type() != 2:
                self.restart_render_event.set()

        if self.is_resized or not self.is_rendered:
            return

        def draw_(texture_id):
            #if False:
        #     if scene.rpr.render_mode in ('WIREFRAME', 'MATERIAL_INDEX',
        #                                  'POSITION', 'NORMAL', 'TEXCOORD'):
                # Draw without color management
            # draw_texture_2d(texture_id, self.viewport_settings.border[0], *self.viewport_settings.border[1])

            # else:
                # Bind shader that converts from scene linear to display space,
                bgl.glEnable(bgl.GL_BLEND)
                bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
                self.cyclesphi_engine.bind_display_space_shader(scene)

                #note this has to draw to region size, not scaled down size
                self._draw_texture(texture_id, *self.viewport_settings.border[0], *self.viewport_settings.border[1])

                self.cyclesphi_engine.unbind_display_space_shader()
                bgl.glDisable(bgl.GL_BLEND)

                #pass

        # if self.is_denoised:
        #     im = None
        #     with self.resolve_lock:
        #         if self.image_filter:
        #             im = self.image_filter.get_data()

        #     if im is not None:
        #         self.gl_texture.set_image(im)
        #         draw_(self.gl_texture.texture_id)
        #         return

        # if self.cyclesphi_context.gl_interop:
        #     with self.resolve_lock:
        #         draw_(self.cyclesphi_context.get_frame_buffer().texture_id)
        #     return

        with self.resolve_lock:
            im = self._get_render_image()

        self.gl_texture.set_image(im)     
        # if _cyclesphi_dll.get_renderengine_type() == 1:           
        #     context.scene.openvr_user_prop.openVrGlRenderer.post_render_scene_left(imL)
        #     context.scene.openvr_user_prop.openVrGlRenderer.post_render_scene_right(imR)

        draw_(self.gl_texture.texture_id)
        #_cyclesphi_dll.draw_texture()
        #self.cyclesphi_context.gl_render(0)
        #self.cyclesphi_context.gl_render(1)
        #self.cyclesphi_context.gl_render(2)

        # Bind shader that converts from scene linear to display space,
        # bgl.glEnable(bgl.GL_BLEND)
        # bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        # self.cyclesphi_engine.bind_display_space_shader(scene)

        # # note this has to draw to region size, not scaled down size
        # self._draw_texture(texture_id, *self.viewport_settings.border[0], *self.viewport_settings.border[1])

        # #_cyclesphi_dll.draw_texture()
        # # if _cyclesphi_dll.get_renderengine_type == 0:
        # #     self.cyclesphi_context.gl_render(2)
        # # else:
        # #     #self.cyclesphi_context.gl_render(0)
        # #     #self.cyclesphi_context.gl_render(1)
        # #     self.cyclesphi_context.gl_render(2)            

        # # border = (0, 0), (self.cyclesphi_context.width, self.cyclesphi_context.height)
        # # self._draw_texture(self.cyclesphi_context.g_textureIds[2], *border[0], *border[1])

        # self.cyclesphi_engine.unbind_display_space_shader()
        # bgl.glDisable(bgl.GL_BLEND)        

        check_gl_error()

class CyclesPhiRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "CYCLESPHI"
    bl_label = "CyclesPhi"
    bl_use_preview = True   

    engine: Engine = None

    # def notify_status(self, info, status):
    #     """ Display export progress status """
    #     wrap_info = textwrap.fill(info, 120)
    #     self.update_stats(status, wrap_info)
        #log(status, wrap_info)

        # requesting blender to call draw()
        #self.tag_redraw()    

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        #self.scene_data = None
        #self.draw_data = None
        self.engine = None
        pass

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        if isinstance(self.engine, ViewportEngine):
            self.engine.stop_render()
            self.engine = None
        pass

    # final render
    def update(self, data, depsgraph):
        """ Called for final render """
        pass    

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        # scene = depsgraph.scene
        # scale = scene.render.resolution_percentage / 100.0
        # self.size_x = int(scene.render.resolution_x * scale)
        # self.size_y = int(scene.render.resolution_y * scale)

        # # Fill the render result with a flat color. The framebuffer is
        # # defined as a list of pixels, each pixel itself being a list of
        # # R,G,B,A values.
        # if self.is_preview:
        #     color = [0.1, 0.2, 0.1, 1.0]
        # else:
        #     color = [0.2, 0.1, 0.1, 1.0]

        # pixel_count = self.size_x * self.size_y
        # rect = [color] * pixel_count

        # # Here we write the pixel values to the RenderResult
        # result = self.begin_result(0, 0, self.size_x, self.size_y)
        # layer = result.layers[0].passes["Combined"]
        # layer.rect = rect
        # self.end_result(result)
        pass

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        region = context.region
        view3d = context.space_data
        scene = depsgraph.scene

        #init resolution
        #_cyclesphi_dll.client_init(region.width, region.height)

        #init camera
        #camera_matrix_world = depsgraph.scene.camera.matrix_world
        #c_camera_matrix_world = (c_float * len(camera_matrix_world))(*camera_matrix_world)
        #_cyclesphi_dll.get_camera(c_camera_matrix_world)
        #c_camera_matrix_world = (c_float * 16)()
        #_cyclesphi_dll.get_camera(depsgraph.scene.camera.as_pointer())

        #depsgraph.scene.camera.matrix_world = c_camera_matrix_world        
        #depsgraph.scene.camera.matrix_world = c_camera_matrix_world

        # if not self.scene_data:
        #     # First time initialization
        #     self.scene_data = []
        #     first_time = True

        #     # Loop over all datablocks used in the scene.
        #     for datablock in depsgraph.ids:
        #         pass
        # else:
        #     first_time = False

        #     # Test which datablocks changed
        #     for update in depsgraph.updates:
        #         print("Datablock updated: ", update.id.name)

        #     # Test if any material was added, removed or changed.
        #     if depsgraph.id_type_updated('MATERIAL'):
        #         print("Materials updated")

        # # Loop over all object instances in the scene.
        # if first_time or depsgraph.id_type_updated('OBJECT'):
        #     for instance in depsgraph.object_instances:
        #         pass


        #self.global_time = time.time()
        #self.global_count = 0

        if self.engine:
            #self.engine.stop_render()
            return

        self.engine = ViewportEngine(self)        
        self.engine.sync(context, depsgraph)

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):       
        # region = context.region
        # scene = depsgraph.scene

        # # update res
        # #_cyclesphi_dll.resize(region.width, region.height)
        # dimensions = (region.width, region.height)
        
        # update camera
        #camera_matrix_world = depsgraph.scene.camera.matrix_world
        #c_camera_matrix_world = (c_float * len(camera_matrix_world))(*camera_matrix_world)
        #c_camera_matrix_world = (c_float * 16)()
        #c_camera_matrix_world = depsgraph.scene.camera.matrix_world
        #_cyclesphi_dll.set_camera(depsgraph.scene.camera.as_pointer())        

        # main loop
        #_cyclesphi_dll.main_loop()

        # Bind shader that converts from scene linear to display space,
        # bgl.glEnable(bgl.GL_BLEND)
        # bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        # self.bind_display_space_shader(scene)

        # if not self.draw_data or self.draw_data.dimensions != dimensions:
        #     self.draw_data = CyclesPhiDrawData(dimensions)

        # #self.draw_data.draw()

        # self.unbind_display_space_shader()
        # bgl.glDisable(bgl.GL_BLEND)

        # self.global_count =  self.global_count + 1

        # if time.time() - self.global_time > 1.0:
        #     #print('FPS: ', self.global_count / (time.time() - self.global_time))
        #     self.notify_status("FPS: %.1f" % (self.global_count / (time.time() - self.global_time)), "Render")
        #     self.global_time = time.time()
        #     self.global_count = 0        

        # #self.notify_status("FPS: %.1f" % (self.global_count / (time.time() - self.global_time)), "Render")
        # self.tag_redraw()

        self.engine.draw(context)

    # view layer AOVs
    def update_render_passes(self, render_scene=None, render_layer=None):
        """
        Update 'Render Layers' compositor node with active render passes info.
        Called by Blender.
        """
        pass        


# class CyclesPhiDrawData:
#     def __init__(self, dimensions):
#         # Generate dummy float image buffer
#         self.dimensions = dimensions
#         width, height = dimensions

#         #self.pixels = numpy.char.array([0, 0, 0, 0] * width * height)
#         #self.pixels = bgl.Buffer(bgl.GL_BYTE, width * height * 4, pixels)

#         # Generate texture
#         # self.texture = bgl.Buffer(bgl.GL_INT, 1)
#         # bgl.glGenTextures(1, self.texture)
#         # bgl.glActiveTexture(bgl.GL_TEXTURE0)
#         # bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
#         # #bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA16F, width, height, 0, bgl.GL_RGBA, bgl.GL_FLOAT, self.pixels)
#         # bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
#         # bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
#         # #bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_NEAREST)
#         # #bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_NEAREST)
#         # # bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP)
#         # # bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP)

#         # bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA, width, height, 0, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, bgl.Buffer(bgl.GL_BYTE, width * height * 4));

#         # bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

#         # # Bind shader that converts from scene linear to display space,
#         # # use the scene's color management settings.
#         # shader_program = bgl.Buffer(bgl.GL_INT, 1)
#         # bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

#         # # Generate vertex array
#         # self.vertex_array = bgl.Buffer(bgl.GL_INT, 1)
#         # bgl.glGenVertexArrays(1, self.vertex_array)
#         # bgl.glBindVertexArray(self.vertex_array[0])

#         # texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
#         # position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

#         # bgl.glEnableVertexAttribArray(texturecoord_location)
#         # bgl.glEnableVertexAttribArray(position_location)

#         # # Generate geometry buffers for drawing textured quad
#         # position = [0.0, 0.0, width, 0.0, width, height, 0.0, height]
#         # position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
#         # texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
#         # texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

#         # self.vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)

#         # bgl.glGenBuffers(2, self.vertex_buffer)
#         # bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[0])
#         # bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
#         # bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

#         # bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[1])
#         # bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
#         # bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

#         # bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)
#         # bgl.glBindVertexArray(0)

#         ########################################################
#         # self.pixel_buffer = bgl.Buffer(bgl.GL_INT, 1)

#         # bgl.glGenBuffers(1, self.pixel_buffer)
#         # bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer[0])
#         # bgl.glBufferData(bgl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, 0, bgl.GL_STREAM_DRAW)
#         # bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, 0)        

#     # def __del__(self):
#     #     #bgl.glDeleteBuffers(1, self.pixel_buffer)
#     #     # bgl.glDeleteBuffers(2, self.vertex_buffer)
#     #     # bgl.glDeleteVertexArrays(1, self.vertex_array)
#     #     # bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
#     #     # bgl.glDeleteTextures(1, self.texture)                

#     def draw(self):
#         # bgl.glActiveTexture(bgl.GL_TEXTURE0)
#         # bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])

#         ########################################################################
#         width, height = self.dimensions

#     #     #bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer[0])
#     #     #gl.glTexSubImage2D(bgl.GL_TEXTURE_2D, 0, 0, 0, width, height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, ctypes.c_void_p(self.pixels.ctypes.data))
#     #     #bgl.glBufferData(bgl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, 0,  bgl.GL_STREAM_DRAW)        
#     #     #ptr_pixels = None #gl.glMapBuffer( bgl.GL_PIXEL_UNPACK_BUFFER,  bgl.GL_WRITE_ONLY)
		
#     #     # #path_tracer(ptr)
#     #     # if ptr_pixels:
#     #     #     c_ptr_pixels = ctypes.cast(ptr_pixels, POINTER(c_char))
#     #     # #     _cyclesphi_dll.get_pixels(c_ptr_pixels)
#     #     #     bgl.glUnmapBuffer(bgl.GL_PIXEL_UNPACK_BUFFER)
#     #     #_cyclesphi_dll.get_pixels(ctypes.c_void_p(self.pixels.ctypes.data))

#     #     # bgl.glBindBuffer(bgl.GL_PIXEL_UNPACK_BUFFER, 0)
#     #     ########################################################################

#     #     # bgl.glBindVertexArray(self.vertex_array[0])
#     #     # bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)
#     #     # bgl.glBindVertexArray(0)
#     #     # bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

#############################
class RenderButtonsPanel:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    # COMPAT_ENGINES must be defined in each subclass, external engines can add themselves here

    @classmethod
    def poll(cls, context):
        return (context.engine in cls.COMPAT_ENGINES)

class RENDER_PT_cyclesphi_server(RenderButtonsPanel, bpy.types.Panel):
    bl_label = "Server"
    COMPAT_ENGINES = {'CYCLESPHI'}

    @classmethod
    def poll(cls, context):
        return (context.engine in cls.COMPAT_ENGINES)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        server_settings = scene.cyclesphi.server_settings

        col = layout.column()
        col.prop(server_settings, "server_name", text="Server")
        col.prop(server_settings, "port_cam", text="Port Cam")
        col.prop(server_settings, "port_data", text="Port Data") 

        col = layout.column()
        col.prop(server_settings, "vrclient_path", text="VRClient")
        col.prop(server_settings, "ssh_command", text="Ssh")        
        col.prop(server_settings, "monitor", text="Monitor")         
        #col.operator("cyclesphi.start_client")      

        #sampling_settings = scene.cyclesphi.sampling_settings

# class RENDER_PT_cyclesphi_rendering(RenderButtonsPanel, bpy.types.Panel):
#     bl_label = "Rendering"
#     COMPAT_ENGINES = {'CYCLESPHI'}

#     @classmethod
#     def poll(cls, context):
#         return (context.engine in cls.COMPAT_ENGINES)

#     def draw(self, context):
#         layout = self.layout
#         layout.use_property_split = True
#         layout.use_property_decorate = False  # No animation.

#         scene = context.scene
#         sampling_settings = scene.cyclesphi.sampling_settings

#         col = layout.column()
#         col.prop(sampling_settings, "width", text="Width")
#         col.prop(sampling_settings, "height", text="Height")

#         col = layout.column()
#         col.prop(sampling_settings, "step_samples", text="Step Samples")

class RENDER_PT_cyclesphi_rendering(RenderButtonsPanel, bpy.types.Panel):
    bl_label = "Rendering"
    COMPAT_ENGINES = {'CYCLESPHI'}

    @classmethod
    def poll(cls, context):
        return (context.engine in cls.COMPAT_ENGINES)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene

        #col = layout.column()
        #col.operator("cyclesphi.connect")
        #col.operator("cyclesphi.disconnect")

        sampling_settings = scene.cyclesphi.sampling_settings

        col = layout.column()        
        col.prop(sampling_settings, "use_viewport", text="Use Viewport")
        col.prop(sampling_settings, "width", text="Width")
        col.prop(sampling_settings, "height", text="Height")

        col = layout.column()
        col.prop(sampling_settings, "step_samples", text="Step Samples")        
        col.prop(sampling_settings, "right_eye", text="Right Eye")

###################################################################################
class VIEW_PT_CYCLESPHI(bpy.types.Panel):
    bl_label = 'CyclesPhi'
    bl_idname = 'VIEW_PT_CYCLESPHI'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CyclesPhi'

    def draw(self, context):
        layout = self.layout
        row = layout.row()

        if context.scene.cyclesphi_data.client_process_id is None:
            row.operator("cyclesphi.start_client")
        else:              
            row.operator("cyclesphi.stop_client")

class VIEW_PT_VRCLIENT(bpy.types.Panel):
    bl_label = 'VRClient'
    bl_idname = 'VIEW_PT_VRCLIENT'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CyclesPhi'

    def draw(self, context):
        layout = self.layout

        scene = context.scene
        server_settings = scene.cyclesphi.server_settings
        sampling_settings = scene.cyclesphi.sampling_settings

        col = layout.column()
        col.prop(server_settings, "server_name", text="Server")
        col.prop(server_settings, "port_cam", text="Port Cam")
        col.prop(server_settings, "port_data", text="Port Data") 

        col = layout.column()
        col.prop(server_settings, "vrclient_path", text="VRClient")
        col.prop(server_settings, "monitor", text="Monitor")

        col = layout.column()
        col.prop(sampling_settings, "step_samples", text="Step Samples") 

        col = layout.column()
        #col.prop(context.region, "width")
        #col.prop(context.region, "height")
        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                for r in a.regions:
                    if r.type == 'WINDOW':
                        col.prop(r, "width")
                        col.prop(r, "height")        

        bpy.context.region.width    

        row = layout.row()

        if context.scene.cyclesphi_data.client_process_id is None:
            row.operator("cyclesphi.start_vrclient")
        else:              
            row.operator("cyclesphi.stop_vrclient")            

###################################################################################

class RENDER_OT_cyclesphi_start_vrclient(bpy.types.Operator):
    """start_vrclient"""
    bl_idname="cyclesphi.start_vrclient"
    bl_label="Start"

    def execute(self, context):

        scene = context.scene
            
        server_settings = scene.cyclesphi.server_settings
        sampling_settings = scene.cyclesphi.sampling_settings

        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                for r in a.regions:
                    if r.type == 'WINDOW':
                        sampling_settings.width = r.width
                        sampling_settings.height = r.height

        import subprocess
        if len(str(server_settings.vrclient_path)) != 0:            
            cmd = '%s %s %d %d %d %d %d %d' % (str(server_settings.vrclient_path), server_settings.server_name, server_settings.port_cam, server_settings.port_data, sampling_settings.step_samples, sampling_settings.width, sampling_settings.height, server_settings.monitor)
            scene.cyclesphi_data.client_process_id = subprocess.Popen(cmd)

        #context.space_data.shading.type = "RENDERED"

        return {'FINISHED'}

class RENDER_OT_cyclesphi_stop_vrclient(bpy.types.Operator):
    """stop_vrclient"""
    bl_idname="cyclesphi.stop_vrclient"
    bl_label="Stop"

    def execute(self, context):

        scene = context.scene  

        import subprocess

        if scene.cyclesphi_data.client_process_id:
            scene.cyclesphi_data.client_process_id.terminate()
            scene.cyclesphi_data.client_process_id = None

        #context.space_data.shading.type = "SOLID"

        return {'FINISHED'}       
###################################################################################
class RENDER_OT_cyclesphi_start_client(bpy.types.Operator):
    """start_client"""
    bl_idname="cyclesphi.start_client"
    bl_label="Start"

    def execute(self, context):

        scene = context.scene
            
        server_settings = scene.cyclesphi.server_settings
        sampling_settings = scene.cyclesphi.sampling_settings      

        import subprocess
        if len(str(server_settings.vrclient_path)) != 0:            
            cmd = '%s %s %d %d %d %d %d %d' % (str(server_settings.vrclient_path), server_settings.server_name, server_settings.port_cam, server_settings.port_data, sampling_settings.step_samples, sampling_settings.width, sampling_settings.height, server_settings.monitor)
            scene.cyclesphi_data.client_process_id = subprocess.Popen(cmd)

        ssh_command = str(server_settings.ssh_command)

        if len(ssh_command) != 0:
            cmd = 'cmd /c ssh %s' % (ssh_command)
            scene.cyclesphi_data.ssh_process_id = subprocess.Popen(cmd, shell=True)

        context.space_data.shading.type = "RENDERED"

        return {'FINISHED'}

class RENDER_OT_cyclesphi_stop_client(bpy.types.Operator):
    """stop_client"""
    bl_idname="cyclesphi.stop_client"
    bl_label="Stop"

    def execute(self, context):

        scene = context.scene  

        import subprocess

        if scene.cyclesphi_data.client_process_id:
            scene.cyclesphi_data.client_process_id.terminate()
            scene.cyclesphi_data.client_process_id = None

        if scene.cyclesphi_data.ssh_process_id:
            scene.cyclesphi_data.ssh_process_id.terminate()
            scene.cyclesphi_data.ssh_process_id = None
        
        if scene.cyclesphi_data.cyclesphi_context:
            scene.cyclesphi_data.cyclesphi_context.client_close_connection()

        context.space_data.shading.type = "SOLID"

        return {'FINISHED'}        

class RENDER_OT_cyclesphi_connect(bpy.types.Operator):
    """Remove an AOV pass"""
    bl_idname="cyclesphi.connect"
    bl_label="Start"

    def execute(self, context):

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'RENDERED'

        return {'FINISHED'}

class RENDER_OT_cyclesphi_disconnect(bpy.types.Operator):
    """Remove an AOV pass"""
    bl_idname="cyclesphi.disconnect"
    bl_label="Stop"

    def execute(self, context):

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'

        return {'FINISHED'}
###################################################################################

# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    # exclude_panels = {
    #     'VIEWLAYER_PT_filter',
    #     'VIEWLAYER_PT_layer_passes',
    # }

    include_panels = {
        'RENDER_PT_color_management',
    }

    panels = []

    panels.append(RENDER_PT_cyclesphi_server)
    #panels.append(RENDER_PT_cyclesphi_sampling)
    panels.append(RENDER_PT_cyclesphi_rendering)

    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ in include_panels:
                panels.append(panel)
    
    return panels

def register():
    # Register the RenderEngine
    bpy.utils.register_class(CyclesPhiRenderEngine)
    bpy.utils.register_class(CyclesPhiServerSettings)
    bpy.utils.register_class(CyclesPhiSamplingSettings)
    bpy.utils.register_class(CyclesPhiRenderSettings)
    bpy.utils.register_class(RENDER_PT_cyclesphi_server)
    #bpy.utils.register_class(RENDER_PT_cyclesphi_sampling)
    bpy.utils.register_class(RENDER_PT_cyclesphi_rendering)
    bpy.utils.register_class(VIEW_PT_CYCLESPHI)
    bpy.utils.register_class(VIEW_PT_VRCLIENT)
    

    #bpy.types.Scene.openvr_user_prop = OpenVrUserProp() 

    bpy.utils.register_class(RENDER_OT_cyclesphi_connect)
    bpy.utils.register_class(RENDER_OT_cyclesphi_disconnect)
    bpy.utils.register_class(RENDER_OT_cyclesphi_start_client)   
    bpy.utils.register_class(RENDER_OT_cyclesphi_stop_client)

    bpy.utils.register_class(RENDER_OT_cyclesphi_start_vrclient)   
    bpy.utils.register_class(RENDER_OT_cyclesphi_stop_vrclient)

    bpy.types.Scene.cyclesphi = bpy.props.PointerProperty(
            name="CyclesPhi Render Settings",
            description="CyclesPhi render settings",
            type=CyclesPhiRenderSettings,
    ) 

    bpy.types.Scene.cyclesphi_data = CyclesPhiData()

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('CYCLESPHI')


def unregister():
    bpy.utils.unregister_class(CyclesPhiRenderEngine)
    bpy.utils.unregister_class(CyclesPhiServerSettings)
    bpy.utils.unregister_class(CyclesPhiSamplingSettings)
    bpy.utils.unregister_class(CyclesPhiRenderSettings)
    bpy.utils.unregister_class(RENDER_PT_cyclesphi_server)
    #bpy.utils.unregister_class(RENDER_PT_cyclesphi_sampling)
    bpy.utils.unregister_class(RENDER_PT_cyclesphi_rendering)
    bpy.utils.unregister_class(VIEW_PT_CYCLESPHI)
    bpy.utils.unregister_class(VIEW_PT_VRCLIENT)
    

    bpy.utils.unregister_class(RENDER_OT_cyclesphi_connect)
    bpy.utils.unregister_class(RENDER_OT_cyclesphi_disconnect)
    bpy.utils.unregister_class(RENDER_OT_cyclesphi_start_client)    
    bpy.utils.unregister_class(RENDER_OT_cyclesphi_stop_client)

    bpy.utils.unregister_class(RENDER_OT_cyclesphi_start_vrclient)    
    bpy.utils.unregister_class(RENDER_OT_cyclesphi_stop_vrclient)    

    delattr(bpy.types.Scene, 'cyclesphi')
    delattr(bpy.types.Scene, 'cyclesphi_data')

    for panel in get_panels():
        if 'CYCLESPHI' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('CYCLESPHI')


if __name__ == "__main__":
    register()
