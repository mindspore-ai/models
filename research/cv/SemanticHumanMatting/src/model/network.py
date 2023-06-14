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

"""The End-to-End of Semantic Human Matting Network"""
import mindspore
from mindspore import Tensor, nn, ops
from mindspore.ops import constexpr
from mindspore.ops import operations as P

from .m_net import M_net
from .t_net import T_mv2_unet


@constexpr
def toTensor(x):
    return Tensor(x, dtype=mindspore.float32)


T_net = T_mv2_unet


class net(nn.Cell):
    """End to end network"""

    def __init__(self, stage=2):
        super(net, self).__init__()
        self.stage = stage

        self.t_net = T_net()
        self.m_net = M_net()

        self.softmax = P.Softmax(axis=1)
        self.split = ops.Split(axis=1, output_num=3)
        self.concat = ops.Concat(axis=1)

    def construct(self, *inputs):
        """
        inputs[0]: clip_img
        inputs[1]: 3-channel trimap
        """
        if self.stage == 0:
            trimap = self.t_net(inputs[0])
            return trimap
        if self.stage == 1:
            trimap_softmax = self.softmax(inputs[1])
            _, fg, unsure = self.split(trimap_softmax)
            m_net_input = self.concat((inputs[0], trimap_softmax))
            # matting
            alpha_r = self.m_net(m_net_input)
            # fusion module
            # paper : alpha_p = fs + us * alpha_r
            alpha_p = fg + unsure * alpha_r
            return alpha_p
        trimap = self.t_net(inputs[0])
        trimap_softmax = self.softmax(trimap)
        _, fg, unsure = self.split(trimap_softmax)
        m_net_input = self.concat((inputs[0], trimap_softmax))
        alpha_r = self.m_net(m_net_input)
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p
