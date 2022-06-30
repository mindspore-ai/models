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

"""
This file include the codes to read .obj files
"""

import numpy as np

def read_obj(filepath):
    vertices = []
    faces = []
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':

            v = [float(x) for x in values[1:4]]
            vertices.append(v)

        elif values[0] == 'f':
            face = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
            faces.append(face)
    vertices = np.array(vertices)
    return vertices, faces


def read_obj_full(filepath):
    vertices = []
    normals = []
    vt_texcoords = []
    faces = []



    material = None
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            # v = map(float, values[1:4])
            v = [float(x) for x in values[1:4]]
            vertices.append(v)
        elif values[0] == 'vn':
            # v = map(float, values[1:4])
            v = [float(x) for x in values[1:4]]
            normals.append(v)
        elif values[0] == 'vt':
            v = [float(x) for x in values[1:3]]
            vt_texcoords.append(v)

        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            texcoords = []
            norms = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) >= 2 and w[1]:
                    texcoords.append(int(w[1]))
                else:
                    texcoords.append(0)
                if len(w) >= 3 and w[2] > 0:
                    norms.append(int(w[2]))
                else:
                    norms.append(0)
            faces.append((face, norms, texcoords, material))
    out_dict = {}
    out_dict['vertices'] = vertices
    out_dict['faces'] = faces
    out_dict['texcoords'] = vt_texcoords
    return out_dict

def write_obj(filepath, vertices, faces):
    with open(filepath, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
