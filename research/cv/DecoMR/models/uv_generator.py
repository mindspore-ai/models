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

import os
from os.path import join
import pickle
import numpy as np
import mindspore
from mindspore import nn, Tensor
import mindspore.ops as ops
from mindspore.ops import constexpr
from models.grid_sample import grid_sample

# '''
# Index_UV_Generator is used to transform mesh and location map
# The verts is in shape (B * V *C)
# The UV map is in shape (B * H * W * C)
# B: batch size;     V: vertex number;   C: channel number
# H: height of uv map;  W: width of uv map
# '''

@constexpr
def generate_Tensor_np(temp):
    return Tensor(temp)

def generate_Tensor(temp):
    return Tensor(temp, dtype=mindspore.float32)

def generate_Tensor_int64(temp):
    return Tensor(temp, dtype=mindspore.int64)

def generate_Tensor_int32(temp):
    return Tensor(temp, dtype=mindspore.int32)

def generate_Tensor_int8(temp):
    return Tensor(temp, dtype=mindspore.int8)


class Index_UV_Generator(nn.Cell):
    def __init__(self, UV_height, UV_width=-1, uv_type='BF', data_dir=None):
        super(Index_UV_Generator, self).__init__()

        self.grid_sample = grid_sample()
        if uv_type == "SMPL":
            obj_file = "smpl_fbx_template.obj"
        elif uv_type == "BF":
            obj_file = "smpl_boundry_free_template.obj"

        self.uv_type = uv_type

        if data_dir is None:
            d = os.path.dirname(__file__)
            data_dir = os.path.join(d, "data", "uv_sampler")
        data_dir = 'data/uv_sampler'
        self.data_dir = data_dir
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        self.obj_file = obj_file
        self.para_file = 'paras_h{:04d}_w{:04d}_{}.npz'.format(self.h, self.w, self.uv_type)

        if not os.path.isfile(join(data_dir, self.para_file)):
            self.process()

        para = np.load(join(data_dir, self.para_file))

        self.v_index = generate_Tensor_int64(para["v_index"])
        self.bary_weights = generate_Tensor(para["bary_weights"])

        self.vt2v = generate_Tensor_int32(para['vt2v'])
        self.vt_count = generate_Tensor(para['vt_count'])


        self.texcoords = generate_Tensor(para['texcoords'])

        self.mask = generate_Tensor_int8(para['mask'].astype('uint8'))

    def get_UV_map(self, verts):
        self.bary_weights = self.bary_weights.astype(verts.dtype)
        self.v_index = self.v_index

        if verts.ndim == 2:
            expand_dims = ops.ExpandDims()
            verts = expand_dims(verts, 0)

        im = verts[:, self.v_index, :]
        bw = self.bary_weights[:, :, None, :]

        squeeze = ops.Squeeze(axis=3)

        im = squeeze(ops.matmul(bw, im))

        return im

    def resample(self, uv_map):
        batch_size, _, _, channel_num = uv_map.shape
        v_num = self.vt_count.shape[0]
        self.texcoords = self.texcoords.astype(uv_map.dtype)
        self.vt2v = self.vt2v
        self.vt_count = self.vt_count.astype(uv_map.dtype)

        shape = (batch_size, -1, -1, -1)
        broadcast_to = ops.BroadcastTo(shape)

        uv_grid = broadcast_to(self.texcoords[None, None, :, :])

        perm = (0, 3, 1, 2)
        transpose = ops.Transpose()
        uv_map = transpose(uv_map, perm)
        vt = self.grid_sample(uv_map, uv_grid)
        squeeze = ops.Squeeze(axis=2)
        vt = squeeze(vt).transpose(0, 2, 1)

        zeros = ops.Zeros()
        v = zeros((batch_size, v_num, channel_num), vt.dtype)

        index_add = ops.IndexAdd(axis=1)
        v = index_add(v, self.vt2v, vt)
        v = v / self.vt_count[None, :, None]
        return v

    # just used for the generation of GT UVmaps
    def forward(self, verts):
        return self.get_UV_map(verts)

# Compute the weight map in UV space according to human body parts.
def cal_uv_weight(sampler, out_path):

    with open('../data/segm_per_v_overlap.pkl', 'rb') as f:
        tmp = pickle.load(f)

    part_names = ['hips',
                  'leftUpLeg',
                  'rightUpLeg',
                  'spine',
                  'leftLeg',
                  'rightLeg',
                  'spine1',
                  'leftFoot',
                  'rightFoot',
                  'spine2',
                  'leftToeBase',
                  'rightToeBase',
                  'neck',
                  'leftShoulder',
                  'rightShoulder',
                  'head',
                  'leftArm',
                  'rightArm',
                  'leftForeArm',
                  'rightForeArm',
                  'leftHand',
                  'rightHand',
                  'leftHandIndex1',
                  'rightHandIndex1']

    part_weight = [1, 5, 5, 1, 5, 5, 1, 25, 25, 1, 25, 25, 2, 1, 1, 2, 5, 5, 5, 5, 25, 25, 25, 25]
    part_weight = generate_Tensor_int64(part_weight)

    zeros = ops.Zeros()
    vert_part = zeros((6890, 24), mindspore.float32)
    for i in range(24):
        key = part_names[i]
        verts = tmp[key]
        vert_part[verts, i] = 1


    squeeze = ops.Squeeze(axis=0)
    part_map = squeeze(sampler.get_UV_map(vert_part))
    part_map = part_map > 0
    weight_map = part_weight[None, None, :].astype("float32") * part_map.astype("float32")
    weight_map = weight_map.max(axis=-1)
    weight_map = weight_map / weight_map.mean()

    np.save(out_path, weight_map.asnumpy())
