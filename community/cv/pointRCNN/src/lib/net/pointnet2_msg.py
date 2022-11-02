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
# This file was copied from project [sshaoshuai][https://github.com/sshaoshuai/PointRCNN]
"""pointnet2 msg"""
from mindspore import nn, Tensor
from src.layer_utils import PointnetSAModuleMSG, PointnetFPModule
from src.lib.config import cfg


def get_model(input_channels=6, use_xyz=True):
    """get model"""
    return Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)


class Pointnet2MSG(nn.Cell):
    """Point2MSG"""
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.CellList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                                    mlps=mlps,
                                    use_xyz=use_xyz,
                                    bn=cfg.RPN.USE_BN))
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.CellList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(
                cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] +
                                 cfg.RPN.FP_MLPS[k]))

    def _break_up_pc(self, pc):
        """break up pc"""
        xyz = pc[..., 0:3]
        features = (pc[..., 3:].swapaxes(1, 2) if pc.shape[-1] > 3 else None)

        return xyz, features

    def construct(self, pointcloud: Tensor):
        """construct function"""
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)

            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i],
                                                   l_features[i - 1],
                                                   l_features[i])

        return l_xyz[0], l_features[0]
