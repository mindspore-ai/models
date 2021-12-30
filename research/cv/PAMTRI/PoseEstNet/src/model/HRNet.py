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
"""HRNet model"""
import mindspore.nn as nn
import mindspore.ops as ops

BN_MOMENTUM = 0.9

class Unuse(nn.Cell):
    """Unuse function"""
    def construct(self, x):
        return x

class Upsample(nn.Cell):
    """Upsample function"""
    def __init__(self, scale_factor=1):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def construct(self, x):
        shape = x.shape
        resize_nearest = ops.ResizeNearestNeighbor((shape[2] * self.scale_factor, shape[3] * self.scale_factor))

        return resize_nearest(x)

class BasicBlock(nn.Cell):
    """BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, plans, stride=1, downSample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, plans,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(plans, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(plans, plans,
                               kernel_size=3,
                               padding=1,
                               pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(plans, momentum=BN_MOMENTUM)
        self.downSample = downSample
        self.stride = stride

    def construct(self, x):
        """BasicBlock construct"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downSample is not None:
            residual = self.downSample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Cell):
    """
    Bottleneck function
    """
    expansion = 4
    def __init__(self, inplanes, plans, stride=1, downSample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, plans,
                               kernel_size=1)
        self.bn1 = nn.BatchNorm2d(plans, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(plans, plans,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(plans, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(plans, plans * self.expansion,
                               kernel_size=1)
        self.bn3 = nn.BatchNorm2d(plans * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downSample = downSample
        self.stride = stride

    def construct(self, x):
        """Bottleneck construct"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downSample is not None:
            residual = self.downSample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Cell):
    """
    HighResolutionModule function
    """
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        """_check_branches"""
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        """_make_one_branch"""
        downsample = None
        if stride != 1 or \
            self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, has_bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            ])
        layers = nn.SequentialCell()
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )

        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return layers

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """_make_branches"""
        branches = nn.CellList()

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return branches

    def _make_fuse_layers(self):
        """_make_fuse_layers"""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = nn.CellList()
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = nn.CellList()
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.SequentialCell([
                            nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                      kernel_size=1,
                                      stride=1),
                            nn.BatchNorm2d(num_inchannels[i]),
                            Upsample(scale_factor=2**(j-i))
                        ])
                    )
                elif j == i:
                    fuse_layer.append(Unuse())
                else:
                    conv3x3s = nn.SequentialCell()
                    for k in range(i-j):
                        if k == i-j-1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.SequentialCell([
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              pad_mode="pad"),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                ])
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.SequentialCell([
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              pad_mode="pad"),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU()
                                ])
                            )
                    fuse_layer.append(conv3x3s)
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        """get_num_inchannels"""
        return self.num_inchannels

    def construct(self, x):
        """HighResolutionModule construct"""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class PoseHighResolutionNet(nn.Cell):
    """PoseHighResolutionNet"""
    def __init__(self, cfg):
        super(PoseHighResolutionNet, self).__init__()
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA

        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=False)


        self.final_layer = nn.Conv2d(pre_stage_channels[0], cfg.MODEL.NUM_JOINTS, \
            kernel_size=extra.FINAL_CONV_KERNEL, stride=1)

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """_make_transition_layer"""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = nn.CellList()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell([
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      pad_mode="pad"),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()
                        ])
                    )
            else:
                conv3x3s = nn.SequentialCell()

                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.SequentialCell(
                            nn.Conv2d(inchannels, outchannels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      pad_mode="pad"),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU()
                        )
                    )
                transition_layers.append(conv3x3s)

        return transition_layers

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """_make_stage"""
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = nn.SequentialCell()
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return modules, num_inchannels

    def construct(self, x):
        """PoseHighResolutionNet construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        x_list.append(self.transition1[0](x))
        x_list.append(self.transition1[1](x))
        y_list = self.stage2(x_list)

        x_list = []
        x_list.append(y_list[0])
        x_list.append(y_list[1])
        x_list.append(self.transition2[0](y_list[-1]))
        y_list = self.stage3(x_list)

        x_list = []
        x_list.append(y_list[0])
        x_list.append(y_list[1])
        x_list.append(y_list[2])
        x_list.append(self.transition3[0](y_list[-1]))
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

def get_pose_net(cfg):
    """get_pose_net"""
    model = PoseHighResolutionNet(cfg)

    return model
