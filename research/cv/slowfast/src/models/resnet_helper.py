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

"""Video models."""

import mindspore.nn as nn

from src.models.common import repeat_3d_tuple


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class BasicTransform(nn.Cell):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner=None,
            num_groups=1,
            stride_1x1=None,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.9,
            norm_module=nn.BatchNorm3d,
            block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the basic block.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Cell): nn.Cell for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, norm_module)

    def _construct(self, dim_in, dim_out, stride, norm_module):
        """Called in Construction."""
        # Tx3x3, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(self.temp_kernel_size, 3, 3),
            stride=(1, stride, stride),
            pad_mode="pad",
            padding=repeat_3d_tuple((int(self.temp_kernel_size // 2), 1, 1)),
            has_bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU()
        # 1x3x3, BN.
        self.b = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            pad_mode="pad",
            padding=repeat_3d_tuple((0, 1, 1)),
            has_bias=False,
        )
        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )

        self.b_bn.transform_final_bn = True

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(nn.Cell):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.9,
            dilation=1,
            norm_module=nn.BatchNorm3d,
            block_idx=0,
        ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Cell): nn.Cell for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
            self,
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        ):
        """Called in construction."""
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(self.temp_kernel_size, 1, 1),
            stride=(1, str1x1, str1x1),
            pad_mode="pad",
            padding=repeat_3d_tuple((int(self.temp_kernel_size // 2), 0, 0)),
            has_bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU()

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            (1, 3, 3),
            stride=(1, str3x3, str3x3),
            pad_mode="pad",
            padding=repeat_3d_tuple((0, dilation, dilation)),
            group=num_groups,
            has_bias=False,
            dilation=(1, dilation, dilation),
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU()

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=repeat_3d_tuple((0, 0, 0)),
            pad_mode="pad",
            has_bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def construct(self, x):
        """Init."""
        # Explicitly construct every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Cell):
    """
    Residual block.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups=1,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.9,
            dilation=1,
            norm_module=nn.BatchNorm3d,
            block_idx=0,
            drop_connect_rate=0.0,
        ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Cell): nn.Cell for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride = stride
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        ):
        """Called in construction."""
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=(1, stride, stride),
                padding=0,
                has_bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        f_x = self.branch2(x)
        if (self.dim_in != self.dim_out) or (self.stride != 1):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class ResStage(nn.Cell):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            stride,
            temp_kernel_sizes,
            num_blocks,
            dim_inner,
            num_groups,
            num_block_temp_kernel,
            nonlocal_inds,
            nonlocal_group,
            nonlocal_pool,
            dilation,
            instantiation="softmax",
            trans_func_name="bottleneck_transform",
            stride_1x1=False,
            inplace_relu=True,
            norm_module=nn.BatchNorm3d,
            drop_connect_rate=0.0,
        ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Cell): nn.Cell for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_block_temp_kernel = num_block_temp_kernel
        self.dilation = dilation
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        )

    def _construct(
            self,
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        ):
        """Called in construction."""
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                trans_func = get_trans_func(trans_func_name)
                # Construct the block.
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                )
                self.insert_child_to_cell(
                    "pathway{}_res{}".format(pathway, i), res_block)
                # no nonlocal in resnet50

    def construct(self, inputs):
        """Init."""
        output = []
        # layer2, NUM_BLOCK_TEMP_KERNEL:3, SPATIAL_DILATIONS:1
        if (self.num_block_temp_kernel[0] == 3 and self.dilation[0] == 1):
            x_slow = inputs[0]
            x_slow = self.pathway0_res0(x_slow)
            x_slow = self.pathway0_res1(x_slow)
            x_slow = self.pathway0_res2(x_slow)
            output.append(x_slow)
            x_fast = inputs[1]
            x_fast = self.pathway1_res0(x_fast)
            x_fast = self.pathway1_res1(x_fast)
            x_fast = self.pathway1_res2(x_fast)
            output.append(x_fast)
        # layer3, NUM_BLOCK_TEMP_KERNEL:4, SPATIAL_DILATIONS:1
        elif (self.num_block_temp_kernel[0] == 4 and self.dilation[0] == 1):
            x_slow = inputs[0]
            x_slow = self.pathway0_res0(x_slow)
            x_slow = self.pathway0_res1(x_slow)
            x_slow = self.pathway0_res2(x_slow)
            x_slow = self.pathway0_res3(x_slow)
            output.append(x_slow)
            x_fast = inputs[1]
            x_fast = self.pathway1_res0(x_fast)
            x_fast = self.pathway1_res1(x_fast)
            x_fast = self.pathway1_res2(x_fast)
            x_fast = self.pathway1_res3(x_fast)
            output.append(x_fast)
        # layer4, NUM_BLOCK_TEMP_KERNEL:6, SPATIAL_DILATIONS:1
        elif (self.num_block_temp_kernel[0] == 6 and self.dilation[0] == 1):
            x_slow = inputs[0]
            x_slow = self.pathway0_res0(x_slow)
            x_slow = self.pathway0_res1(x_slow)
            x_slow = self.pathway0_res2(x_slow)
            x_slow = self.pathway0_res3(x_slow)
            x_slow = self.pathway0_res4(x_slow)
            x_slow = self.pathway0_res5(x_slow)
            output.append(x_slow)
            x_fast = inputs[1]
            x_fast = self.pathway1_res0(x_fast)
            x_fast = self.pathway1_res1(x_fast)
            x_fast = self.pathway1_res2(x_fast)
            x_fast = self.pathway1_res3(x_fast)
            x_fast = self.pathway1_res4(x_fast)
            x_fast = self.pathway1_res5(x_fast)
            output.append(x_fast)
        elif (self.num_block_temp_kernel[0] == 3 and self.dilation[0] == 2):
            x_slow = inputs[0]
            x_slow = self.pathway0_res0(x_slow)
            x_slow = self.pathway0_res1(x_slow)
            x_slow = self.pathway0_res2(x_slow)
            output.append(x_slow)
            x_fast = inputs[1]
            x_fast = self.pathway1_res0(x_fast)
            x_fast = self.pathway1_res1(x_fast)
            x_fast = self.pathway1_res2(x_fast)
            output.append(x_fast)

        return output
