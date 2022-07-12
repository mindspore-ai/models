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

"""Generic_UNet class"""

from copy import deepcopy

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import initializer, HeNormal
import numpy as np


from src.nnunet.network_architecture.initialization import InitWeights_He
from src.nnunet.network_architecture.neural_network import SegmentationNetwork
from src.nnunet.utilities.nd_softmax import softmax_helper


class ConvDropoutNormNonlin(nn.SequentialCell):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        """init class"""
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'alpha': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.9}  # mindspore need sub momentum
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'has_bias': True,
                           'pad_mode': 'pad'}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instancenorm = self.norm_op(output_channels * 9, **self.norm_op_kwargs)  # for task04
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def construct(self, x):
        """construct network"""
        shape = ops.Shape()
        x = self.conv(x)
        n, c, d, w, h = shape(x)
        x = ops.Reshape()(x, (1, n * c, d, h, w))
        x = self.instancenorm(x)
        x = ops.Reshape()(x, (n, c, d, h, w))

        return self.lrelu(x)


class ConvDropoutNormNonlin2D(nn.SequentialCell):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        """init class"""
        super(ConvDropoutNormNonlin2D, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'alpha': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.9}  # mindspore need sub momentum
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'has_bias': True,
                           'pad_mode': 'pad'}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = nn.BatchNorm2d

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instancenorm = self.norm_op(output_channels * 366,
                                         **self.norm_op_kwargs)  # for task04
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def construct(self, x):
        """construct network"""
        shape = ops.Shape()
        x = self.conv(x)
        n, c, h, w = shape(x)
        x = ops.Reshape()(x, (1, n * c, h, w))
        x = self.instancenorm(x)
        x = ops.Reshape()(x, (n, c, h, w))

        return self.lrelu(x)


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def construct(self, x):
        """construct network"""
        shape = ops.Shape()
        x = self.conv(x)
        x = self.lrelu(x)
        n, c, d, w, h = shape(x)
        x = ops.Reshape()(x, (1, n * c, d, h, w))
        x = self.instnorm(x)

        return ops.Reshape()(x, (n, c, d, h, w))


class StackedConvLayers(nn.SequentialCell):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        """
            stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack.
            The other parameters affect all layers
        """

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'alpha': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'has_bias': True,
                           'pad_mode': 'pad'}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()

        self.cell_list = ([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                                       self.conv_kwargs_first_conv,
                                       self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                       self.nonlin, self.nonlin_kwargs)] +
                          [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                                       self.conv_kwargs,
                                       self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                       self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)])
        self.blocks = nn.SequentialCell(self.cell_list)

    def construct(self, x):
        """construct network"""
        return self.blocks(x)


def print_module_training_status(module):
    """print module training status"""
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Dropout3d, nn.Dropout2d,
                           nn.Dropout, nn.InstanceNorm3d, nn.InstanceNorm2d,
                           nn.InstanceNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
        print(str(module), module.training)


class Upsample(nn.Cell):
    """upsample class"""

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        """init upsample"""
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def construct(self, x):
        """construct network"""
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet(SegmentationNetwork):
    """Generic UNet Parameters"""
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'alpha': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'has_bias': True, "pad_mode": "pad"}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.Conv2dTranspose
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
            basic_block = ConvDropoutNormNonlin2D


        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = ops.MaxPool3D
            transpconv = nn.Conv3dTranspose
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = tuple(pool_op_kernel_sizes[d - 1])  # Mindspore need 6 tuple
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = tuple(self.conv_kernel_sizes[d])
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d][0]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = tuple(pool_op_kernel_sizes[-1])
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = tuple(self.conv_kernel_sizes[num_pool])
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool][0]
        self.conv_blocks_context.append(nn.SequentialCell(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(
                    transpconv(nfeatures_from_down, nfeatures_from_skip, tuple(pool_op_kernel_sizes[-(u + 1)]),
                               tuple(pool_op_kernel_sizes[-(u + 1)]), has_bias=False))

            self.conv_kwargs['kernel_size'] = tuple(self.conv_kernel_sizes[- (u + 1)])
            self.conv_kwargs['padding'] = (self.conv_pad_sizes[- (u + 1)][0])
            self.conv_blocks_localization.append(nn.SequentialCell(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append((conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                             kernel_size=1, stride=1, padding=0, dilation=1, group=1,
                                             has_bias=seg_output_use_bias)))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(self.g)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.CellList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.CellList(self.conv_blocks_context)
        self.td = nn.CellList(self.td)
        self.tu = nn.CellList(self.tu)
        self.seg_outputs = nn.CellList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.CellList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            if isinstance(self.weightInitializer, InitWeights_He):
                self.init_weights(self.cells())

    def init_weights(self, cell):
        """init_weights"""
        for op_cell in cell:

            if isinstance(op_cell, nn.SequentialCell):
                self.init_weights(op_cell)
            else:
                if isinstance(op_cell, (nn.Conv3d, nn.Conv2d, nn.Conv2dTranspose, nn.Conv3dTranspose)):
                    op_cell.weight.set_data = initializer(HeNormal(), op_cell.weight.shape, ms.float32)
                    if op_cell.bias is not None:
                        op_cell.bias.set_data = initializer(0, op_cell.bias.shape, ms.float32)

    def g(self, x):
        """same result as lambda x:x"""
        return x

    def construct(self, x):
        """construct network"""
        skips = []
        seg_outputs = []

        x = self.conv_blocks_context[0](x)
        skips.append(x)
        x = self.conv_blocks_context[1](x)
        skips.append(x)
        x = self.conv_blocks_context[2](x)
        skips.append(x)
        x = self.conv_blocks_context[3](x)
        x = self.tu[0](x)
        x = ops.Concat(1)((x, skips[-(0 + 1)]))
        x = self.conv_blocks_localization[0](x)
        x0 = self.seg_outputs[0](x)

        seg_outputs.append(self.final_nonlin(x0))

        x = self.tu[1](x)
        x = ops.Concat(1)((x, skips[-(1 + 1)]))
        x = self.conv_blocks_localization[1](x)
        x1 = self.seg_outputs[1](x)
        seg_outputs.append(self.final_nonlin(x1))
        x = self.tu[2](x)
        x = ops.Concat(1)((x, skips[-(2 + 1)]))
        x = self.conv_blocks_localization[2](x)
        x2 = self.seg_outputs[2](x)
        seg_outputs.append(self.final_nonlin(x2))

        if self._deep_supervision and self.do_ds:
            return ([seg_outputs[-1]] + [seg_outputs[1], seg_outputs[0]])

        if not self._deep_supervision and not self.do_ds:
            return seg_outputs[-1]
        return None

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (
                npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp
