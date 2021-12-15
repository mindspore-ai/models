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
"""delf model"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal


class _IdentityBlock(nn.Cell):
    """Identity Block"""

    def __init__(self, kernel_size, filters, stage, block):
        super(_IdentityBlock, self).__init__(auto_prefix=False)

        filters1, filters2, filters3, filters4 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = nn.Conv2d(filters4, filters1, 1,
                               pad_mode='valid', has_bias=True)
        self.conv1.update_parameters_name(conv_name_base + '2a'+'.')

        self.conv19BatchNorm = nn.BatchNorm2d(
            num_features=filters1, eps=0.001, momentum=0.99)
        self.conv19BatchNorm.update_parameters_name(bn_name_base + '2a'+'.')

        self.conv2 = nn.Conv2d(
            filters1, filters2, kernel_size, pad_mode='same', has_bias=True)
        self.conv2.update_parameters_name(conv_name_base + '2b'+'.')

        self.conv29BatchNorm = nn.BatchNorm2d(
            num_features=filters2, eps=0.001, momentum=0.99)
        self.conv29BatchNorm.update_parameters_name(bn_name_base + '2b'+'.')

        self.conv3 = nn.Conv2d(filters2, filters3, 1,
                               pad_mode='valid', has_bias=True)
        self.conv3.update_parameters_name(conv_name_base + '2c'+'.')

        self.conv39BatchNorm = nn.BatchNorm2d(
            num_features=filters3, eps=0.001, momentum=0.99)
        self.conv39BatchNorm.update_parameters_name(bn_name_base + '2c'+'.')

        self.relu = nn.ReLU()

    def construct(self, input_tensor):
        """construct"""
        x = self.conv1(input_tensor)
        x = self.conv19BatchNorm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv29BatchNorm(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.conv39BatchNorm(x)

        x += input_tensor
        return self.relu(x)


class _ConvBlock(nn.Cell):
    """Conv Block"""

    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 strides=(2, 2)):
        super(_ConvBlock, self).__init__(auto_prefix=False)
        filters1, filters2, filters3, filters4 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = nn.Conv2d(filters4, filters1, 1,
                               stride=strides, pad_mode='valid', has_bias=True)
        self.conv1.update_parameters_name(conv_name_base + '2a'+'.')

        self.conv19BatchNorm = nn.BatchNorm2d(
            num_features=filters1, eps=0.001, momentum=0.99)
        self.conv19BatchNorm.update_parameters_name(bn_name_base + '2a'+'.')

        self.conv2 = nn.Conv2d(
            filters1, filters2, kernel_size, pad_mode='same', has_bias=True)
        self.conv2.update_parameters_name(conv_name_base + '2b'+'.')

        self.conv29BatchNorm = nn.BatchNorm2d(
            num_features=filters2, eps=0.001, momentum=0.99)
        self.conv29BatchNorm.update_parameters_name(bn_name_base + '2b'+'.')

        self.conv3 = nn.Conv2d(filters2, filters3, 1,
                               pad_mode='valid', has_bias=True)
        self.conv3.update_parameters_name(conv_name_base + '2c'+'.')

        self.conv39BatchNorm = nn.BatchNorm2d(
            num_features=filters3, eps=0.001, momentum=0.99)
        self.conv39BatchNorm.update_parameters_name(bn_name_base + '2c'+'.')

        self.shortcut = nn.Conv2d(
            filters4, filters3, 1, stride=strides, pad_mode='valid', has_bias=True)
        self.shortcut.update_parameters_name(conv_name_base + '1'+'.')

        self.shortcut9BatchNorm = nn.BatchNorm2d(
            num_features=filters3, eps=0.001, momentum=0.99)
        self.shortcut9BatchNorm.update_parameters_name(bn_name_base + '1'+'.')

        self.relu = nn.ReLU()

    def construct(self, input_tensor):
        """construct"""
        x = self.conv1(input_tensor)
        x = self.conv19BatchNorm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv29BatchNorm(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.conv39BatchNorm(x)

        shortcut = self.shortcut(input_tensor)
        shortcut = self.shortcut9BatchNorm(shortcut)

        x += shortcut
        return self.relu(x)


class ResNet50(nn.Cell):
    """ResNet50"""

    def __init__(self):
        super(ResNet50, self).__init__(auto_prefix=False)

        def conv_block(filters, stage, block, strides=(2, 2)):
            return _ConvBlock(
                3,
                filters,
                stage=stage,
                block=block,
                strides=strides)

        def id_block(filters, stage, block):
            return _IdentityBlock(
                3, filters, stage=stage, block=block)

        self.conv1 = nn.Conv2d(
            3, 64, 7, stride=2, pad_mode='same', has_bias=True)
        self.conv1.update_parameters_name('conv1.')

        self.bn_conv1 = nn.BatchNorm2d(
            num_features=64, eps=0.001, momentum=0.99)
        self.bn_conv1.update_parameters_name('bn_conv1.')

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.block19unit_19bottleneck_v1 = conv_block(
            [64, 64, 256, 64], stage=2, block='a', strides=(1, 1))
        self.block19unit_29bottleneck_v1 = id_block(
            [64, 64, 256, 256], stage=2, block='b')
        self.block19unit_39bottleneck_v1 = id_block(
            [64, 64, 256, 256], stage=2, block='c')

        self.block29unit_19bottleneck_v1 = conv_block(
            [128, 128, 512, 256], stage=3, block='a')
        self.block29unit_29bottleneck_v1 = id_block(
            [128, 128, 512, 512], stage=3, block='b')
        self.block29unit_39bottleneck_v1 = id_block(
            [128, 128, 512, 512], stage=3, block='c')
        self.block29unit_49bottleneck_v1 = id_block(
            [128, 128, 512, 512], stage=3, block='d')

        self.block39unit_19bottleneck_v1 = conv_block(
            [256, 256, 1024, 512], stage=4, block='a')
        self.block39unit_29bottleneck_v1 = id_block(
            [256, 256, 1024, 1024], stage=4, block='b')
        self.block39unit_39bottleneck_v1 = id_block(
            [256, 256, 1024, 1024], stage=4, block='c')
        self.block39unit_49bottleneck_v1 = id_block(
            [256, 256, 1024, 1024], stage=4, block='d')
        self.block39unit_59bottleneck_v1 = id_block(
            [256, 256, 1024, 1024], stage=4, block='e')
        self.block39unit_69bottleneck_v1 = id_block(
            [256, 256, 1024, 1024], stage=4, block='f')

        self.subsampling_layer = nn.MaxPool2d(
            kernel_size=1, stride=2, pad_mode="valid")

        self.l5a = conv_block([512, 512, 2048, 1024],
                              stage=5, block='a', strides=(1, 1))

        self.l5b = id_block([512, 512, 2048, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048, 2048], stage=5, block='c')

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7, pad_mode="valid")

        self.global_pooling = ops.ReduceMean(keep_dims=False)

        self.relu = nn.ReLU()
        self.iden = ops.Identity()

    def construct(self, inputs):
        """Building the ResNet50 model.

        Args:
            inputs: Images to compute features for.

        Returns:
            Tensor with featuremap.
        """

        x = self.conv1(inputs)
        x = self.bn_conv1(x)
        x = self.relu(x)

        x = self.max_pool(x)

        # Block 1 (equivalent to "conv2" in Resnet paper).
        x = self.block19unit_19bottleneck_v1(x)
        x = self.block19unit_29bottleneck_v1(x)
        x = self.block19unit_39bottleneck_v1(x)

        # Block 2 (equivalent to "conv3" in Resnet paper).
        x = self.block29unit_19bottleneck_v1(x)
        x = self.block29unit_29bottleneck_v1(x)
        x = self.block29unit_39bottleneck_v1(x)
        x = self.block29unit_49bottleneck_v1(x)

        # Block 3 (equivalent to "conv4" in Resnet paper).
        x = self.block39unit_19bottleneck_v1(x)
        x = self.block39unit_29bottleneck_v1(x)
        x = self.block39unit_39bottleneck_v1(x)
        x = self.block39unit_49bottleneck_v1(x)
        x = self.block39unit_59bottleneck_v1(x)
        x = self.block39unit_69bottleneck_v1(x)

        x = self.subsampling_layer(x)
        block3 = x

        x = self.l5a(x)
        x = self.l5b(x)
        x = self.l5c(x)

        x = self.avg_pool(x)

        x = self.global_pooling(x, (2, 3))
        return x, block3


class AttentionModel(nn.Cell):
    """
    attention net
    """

    def __init__(self, num_channel=1024):
        super(AttentionModel, self).__init__()

        conv1_weight = initializer(
            Normal(0.05, 0.0), [512, 1024, 1, 1], mindspore.float32)
        conv1_bias = initializer(Normal(1.5, -1.0), [512], mindspore.float32)
        conv2_weight = initializer(
            Normal(0.5, 1.0), [1, 512, 1, 1], mindspore.float32)
        conv2_bias = initializer(-0.6666, [1], mindspore.float32)
        bn_beta = initializer(Normal(0.17, -0.15), [512], mindspore.float32)

        self.conv1_needl2 = nn.Conv2d(
            num_channel, 512, 1, pad_mode='same', has_bias=True, weight_init=conv1_weight, bias_init=conv1_bias)
        self.bn1 = nn.BatchNorm2d(
            num_features=512, eps=0.001, momentum=0.99, beta_init=bn_beta)
        self.conv2_needl2 = nn.Conv2d(
            512, 1, 1, pad_mode='same', has_bias=True, weight_init=conv2_weight, bias_init=conv2_bias)

        self.relu = nn.ReLU()
        self.activation_layer = ops.Softplus()

        self.l2norm = ops.L2Normalize(axis=1)
        self.reducemean = ops.ReduceMean(keep_dims=False)
        self.mul = ops.Mul()

    def construct(self, inputs, targets=None):
        """construct"""
        x = self.conv1_needl2(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        score = self.conv2_needl2(x)
        prob = self.activation_layer(score)

        if targets is None:
            targets = inputs

        targets = self.l2norm(targets)
        feat = self.reducemean(self.mul(targets, prob), (2, 3))

        return feat, prob, score


class Model(nn.Cell):
    """delf model"""

    def __init__(self, state='tuning', num_classes=81313):
        super(Model, self).__init__(auto_prefix=False)
        self.backbone = ResNet50()
        self.state = state
        if self.state == "tuning":
            self.desc_classification_clean = nn.Dense(
                2048, num_classes, weight_init='XavierUniform', activation=None)

            self.desc_classification_clean.update_parameters_name('desc.')
        elif self.state == "attn":
            self.desc_classification_clean = nn.Dense(
                2048, num_classes, weight_init='XavierUniform', activation=None)

            self.desc_classification_clean.update_parameters_name('desc.')

            self.attention = AttentionModel(1024)
            self.attention.update_parameters_name('attention.')

            self.attn_classification_clean = nn.Dense(
                1024, num_classes, weight_init='XavierUniform', activation=None)

            self.attn_classification_clean.update_parameters_name('attn.')
        else:
            self.attention = AttentionModel(1024)
            self.attention.update_parameters_name('attention.')

    def construct(self, images):
        """construct"""
        desc_prelogits, stop_block3 = self.backbone(images)
        if self.state == "tuning":
            desc_logits = self.desc_classification_clean(desc_prelogits)
            result = desc_logits
        elif self.state == "attn":
            desc_logits = self.desc_classification_clean(desc_prelogits)
            attn_prelogits, attn_scores, _ = self.attention(stop_block3)
            attn_logits = self.attn_classification_clean(attn_prelogits)
            result = (attn_logits, desc_logits)
        else:
            _, attn_scores, _ = self.attention(stop_block3)
            result = (attn_scores, stop_block3)
        return result
