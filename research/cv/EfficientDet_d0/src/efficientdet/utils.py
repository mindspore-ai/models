# Copyright 2021 Huawei Technologies Co., Ltd
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
""" utils for efficientdet """
import mindspore
import mindspore.nn as nn
from mindspore import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import ops as op


class BoxEncoder(nn.Cell):
    """ encode box to xywh """
    def __init__(self):
        super().__init__()
        self.squeeze = op.Squeeze()
        self.concat = op.Concat(2)
        self.min = Tensor(1, mstype.float32)
        self.max = Tensor(512, mstype.float32)

    def construct(self, box):
        """ pylint """

        a = (box[:, :, 0:1:1] + box[:, :, 2:3:1]) / 2    # center_x
        b = (box[:, :, 1:2:1] + box[:, :, 3:4:1]) / 2   # center_y
        c = (box[:, :, 2:3:1] - box[:, :, 0:1:1])
        d = (box[:, :, 3:4:1] - box[:, :, 1:2:1])
        bboxes = self.concat((a, b, c, d))     # [x,y,w,h]

        return bboxes


class Anchors(nn.Cell):
    """ Anchors Generator """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [8, 16, 32, 64, 128]
        self.scales = [1, 1.2599211, 1.587401]
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

        self.expandDims = op.ExpandDims()
        self.linspace = op.LinSpace()
        self.tot = op.ScalarToTensor()
        self.abs = op.Abs()
        self.meshgrid = op.Meshgrid(indexing="xy")
        self.scalar_cast = op.ScalarCast()
        self.tile = op.Tile()
        self.box_enchor = BoxEncoder()
        self.reshape = op.Reshape()

        self._range = []
        for i in range(5):
            self._range.append(op.range(Tensor(self.strides[i] / 2, mstype.int32),
                                        Tensor(512, mstype.int32),
                                        Tensor(self.strides[i], mstype.int32)))


    def construct(self, image):
        """ pylint """

        batch_size = image.shape[0]

        index = [0, 1, 2, 3, 4]

        boxes_all = ()
        for i, stride in zip(index, self.strides):
            boxes_level = ()

            for scale in self.scales:
                for ratio in self.ratios:

                    scale = self.scalar_cast(scale, mindspore.float64)
                    base_anchor_size = self.anchor_scale * stride * scale
                    base_anchor_size = self.scalar_cast(base_anchor_size, mindspore.float32)
                    anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                    anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                    x = self._range[i]()
                    y = self._range[i]()

                    inputs = (x, y)

                    xv, yv = self.meshgrid(inputs)

                    xv = xv.view(-1)
                    yv = yv.view(-1)

                    boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                       yv + anchor_size_y_2, xv + anchor_size_x_2))
                    boxes = np.swapaxes(boxes, 0, 1)
                    boxes_level += (np.expand_dims(boxes, 1),)

            boxes_level = np.concatenate(boxes_level, 1)
            boxes_all += (self.reshape(boxes_level, (-1, 4)),)

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = anchor_boxes.astype(mindspore.float32)
        anchor_boxes = self.expandDims(anchor_boxes, 0)
        anchor_boxes = self.tile(anchor_boxes, (batch_size, 1, 1))

        return anchor_boxes
