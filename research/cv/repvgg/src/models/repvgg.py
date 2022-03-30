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
"""ConvNext Model Define"""

import copy
import os

import numpy as np
from mindspore import Tensor, nn, ops
from mindspore import save_checkpoint
from mindspore.common import initializer as weight_init

from src.models.layers.group_conv import GroupConv2d
from src.models.layers.identity import Identity

if os.getenv("DEVICE_TARGET") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")

class SEBlock(nn.Cell):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              has_bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            has_bias=True)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_channels = input_channels

    def construct(self, inputs):
        B = inputs.shape[0]
        x = ops.ReduceMean(False)(inputs, [2, 3])
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = self.sigmoid(x)
        x = ops.Reshape()(x, (B, -1, 1, 1))
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, group=1):
    device_target = os.getenv("DEVICE_TARGET")
    if device_target == "GPU":
        conv2d = nn.Conv2d
    else:
        conv2d = GroupConv2d
    cell = nn.SequentialCell(
        [
            conv2d(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, padding=padding, group=group, pad_mode="pad",
                   has_bias=False),
            BatchNorm2d(num_features=out_channels)
        ]
    )
    return cell


class RepVGGBlock(nn.Cell):
    """RepVGGBlock"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, group=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.group = group
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = Identity()

        if deploy:
            device_target = os.getenv("DEVICE_TARGET")
            if device_target == "GPU":
                conv2d = nn.Conv2d
            else:
                conv2d = GroupConv2d
            self.rbr_reparam = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, group=group, bias=True,
                                      padding_mode=padding_mode)
        else:
            self.rbr_reparam = None
            self.rbr_identity = BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, group=group)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, group=group)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def construct(self, inputs):
        if self.rbr_reparam is not None:
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight

        t3 = self.rbr_dense.bn.weight / (
            ops.Sqrt()((self.rbr_dense.bn.moving_variance + self.rbr_dense.bn.eps)))
        t3 = ops.Reshape()(t3, (-1, 1, 1, 1))

        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.moving_variance + self.rbr_1x1.bn.eps).sqrt()))
        t1 = ops.Reshape()(t1, (-1, 1, 1, 1))

        l2_loss_circle = ops.ReduceSum()(K3 ** 2) - ops.ReduceSum()(K3[:, :, 1:2, 1:2] ** 2)
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = ops.ReduceSum()(eq_kernel ** 2 / (t3 ** 2 + t1 ** 2))
        return l2_loss_eq_kernel + l2_loss_circle

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return ops.Pad(((1, 1), (1, 1)))(kernel1x1)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.SequentialCell):
            kernel = branch.conv.weight
            moving_mean = branch.bn.moving_mean
            moving_variance = branch.bn.moving_variance
            gamma = branch.bn.gamma
            beta = branch.bn.beta
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.BatchNorm2d, nn.SyncBatchNorm))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.group
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = Tensor(kernel_value, dtype=branch.weight.dtype)
            kernel = self.id_tensor
            moving_mean = branch.moving_mean
            moving_variance = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.eps
        std = ops.Sqrt()(moving_variance + eps)
        t = ops.Reshape()(gamma / std, (-1, 1, 1, 1))
        return kernel * t, beta - moving_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     group=self.rbr_dense.conv.group, has_bias=True, pad_mode="pad")
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Cell):
    """RepVGG Build"""
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_group_map=None,
                 deploy=False, use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_group_map = override_group_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_group_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = ops.ReduceMean(False)
        self.linear = nn.Dense(int(512 * width_multiplier[3]), num_classes)
        self.init_weights()

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            cur_group = self.override_group_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=s, padding=1, group=cur_group, deploy=self.deploy,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.SequentialCell(blocks)

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def construct(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x, (2, 3))
        x = self.linear(x)
        return x


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False, num_classes=1000):
    """create_RepVGG_A0"""
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_group_map=None, deploy=deploy)


def create_RepVGG_A1(deploy=False, num_classes=1000):
    """create_RepVGG_A1"""
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_group_map=None, deploy=deploy)


def create_RepVGG_A2(deploy=False, num_classes=1000):
    """create_RepVGG_A2"""
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_group_map=None, deploy=deploy)


def create_RepVGG_B0(deploy=False, num_classes=1000):
    """create_RepVGG_B0"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_group_map=None, deploy=deploy)


def create_RepVGG_B1(deploy=False, num_classes=1000):
    """create_RepVGG_B1"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_group_map=None, deploy=deploy)


def create_RepVGG_B1g2(deploy=False, num_classes=1000):
    """create_RepVGG_B1g2"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_group_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(deploy=False, num_classes=1000):
    """create_RepVGG_B2"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_group_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False, num_classes=1000):
    """create_RepVGG_B2"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_group_map=None, deploy=deploy)


def create_RepVGG_B2g2(deploy=False, num_classes=1000):
    """create_RepVGG_B3"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_group_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(deploy=False, num_classes=1000):
    """create_RepVGG_B3"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_group_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False, num_classes=1000):
    """create_RepVGG_B3"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_group_map=None, deploy=deploy)


def create_RepVGG_B3g2(deploy=False, num_classes=1000):
    """create_RepVGG_B3g2"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_group_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(deploy=False, num_classes=1000):
    """create_RepVGG_B3g4"""
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_group_map=g4_map, deploy=deploy)


def create_RepVGG_D2se(deploy=False, num_classes=1000):
    """create_RepVGG_D2se"""
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_group_map=None, deploy=deploy, use_se=True)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-D2se': create_RepVGG_D2se,
}


def get_RepVGG_func_by_name(name):
    """get_RepVGG_func_by_name"""
    return func_dict[name]


def repvgg_model_convert(model: nn.Cell, save_path=None, do_copy=True):
    """repvgg_model_convert"""
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        save_checkpoint(model.parameters_and_names(), save_path)
    return model
