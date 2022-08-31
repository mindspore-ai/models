# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from chumpy.ch import Ch
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight

# Rotate the points by a specified angle.
def rotateY(points, angle):
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)

class Renderer:
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353],
                       'blue': [0.65098039, 0.74117647, 0.85882353], 'green': [180.0/255.0, 238.0/255.0, 180.0/255],
                       'tan': [1.0, 218.0/255, 185.0/255]}
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5, height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot, t=camera_t, f=focal_length * np.ones(2),
                                             c=camera_center, k=np.zeros(5))

        dist = np.abs(self.renderer.camera.t.r[2] - np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far, 'width': width, 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1., 1., 1.]

        self.renderer.set(v=vertices, f=faces, vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc

        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(f=self.renderer.f, v=self.renderer.v,
                                                num_verts=self.renderer.v.shape[0],
                                                light_pos=rotateY(np.array([-200, -100, -100]), yrot), vc=albedo,
                                                light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(f=self.renderer.f, v=self.renderer.v,
                                                 num_verts=self.renderer.v.shape[0],
                                                 light_pos=rotateY(np.array([800, 10, 300]), yrot), vc=albedo,
                                                 light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(f=self.renderer.f, v=self.renderer.v,
                                                 num_verts=self.renderer.v.shape[0],
                                                 light_pos=rotateY(np.array([-500, 500, 1000]), yrot), vc=albedo,
                                                 light_color=np.array([.7, .7, .7]))

        return self.renderer.r


def render_IUV(img, vertices, camera, renderer, color='pink', focal_length=1000):
    """
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] + 1e-9)])

    rend_img = renderer.render(vertices, camera_t=camera_t, img=img, use_bg=True,
                               focal_length=focal_length, body_color=color)

    return rend_img


class UVRenderer:
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None, tex=None, vt=None, ft=None):
        self.colors = {'pink': [.9, .7, .7], 'blue': [0.65098039, 0.74117647, 0.85882353]}
        self.width = width
        self.height = height
        self.faces = faces
        self.tex = tex
        self.vt = vt
        self.ft = ft
        self.renderer = TexturedRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None, use_bg=False,
               bg_color=(0.0, 0.0, 0.0), body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):

        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot, t=camera_t, f=focal_length * np.ones(2),
                                             c=camera_center, k=np.zeros(5))

        dist = np.abs(self.renderer.camera.t.r[2] - np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0,
                                 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1., 1., 1.]

        self.renderer.set(v=vertices, f=faces, vt=self.vt, ft=self.ft,
                          vc=color, bgcolor=np.ones(3), texture_image=self.tex)

        self.renderer.vc = Ch(np.ones([6890, 3]))

        _ = self.renderer.r
        out = self.renderer.texcoord_image
        return out
