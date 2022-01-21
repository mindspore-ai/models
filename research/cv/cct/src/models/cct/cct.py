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
"""cct model"""
import mindspore.common.initializer as weight_init
import mindspore.nn as nn

from src.models.cct.tokenizer import Tokenizer
from src.models.cct.transformers import TransformerClassifier
from src.models.cct.var_init import KaimingNormal


class CCT(nn.Cell):
    """CCT Model"""

    def __init__(
            self,
            img_size=224,
            embedding_dim=768,
            n_input_channels=3,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            num_layers=14,
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=1000,
            positional_embedding='learnable'):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=n_input_channels,
                height=img_size,
                width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding)
        self.init_weights()

    def construct(self, x):
        x = self.tokenizer(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    weight_init.initializer(
                        KaimingNormal(
                            mode='fan_in'),
                        cell.weight.shape,
                        cell.weight.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    weight_init.initializer(
                        weight_init.TruncatedNormal(
                            sigma=0.02),
                        cell.weight.shape,
                        cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(
                            weight_init.Zero(),
                            cell.bias.shape,
                            cell.bias.dtype))


def _cct(arch,
         num_layers,
         num_heads,
         mlp_ratio,
         embedding_dim,
         kernel_size=3,
         stride=None,
         padding=None,
         **kwargs):
    """get cct model with parameters"""
    print(f'=> using arch: {arch}')
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **kwargs)
    return model


def cct_2(arch, **kwargs):
    """cct_2"""
    return _cct(
        arch,
        num_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=128,
        **kwargs)


def cct_4(arch, **kwargs):
    """cct_4"""
    return _cct(
        arch,
        num_layers=4,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=128,
        **kwargs)


def cct_6(arch, **kwargs):
    """cct_6"""
    return _cct(
        arch,
        num_layers=6,
        num_heads=4,
        mlp_ratio=2,
        embedding_dim=256,
        **kwargs)


def cct_7(arch, **kwargs):
    """cct_7"""
    return _cct(
        arch,
        num_layers=7,
        num_heads=4,
        mlp_ratio=2,
        embedding_dim=256,
        **kwargs)


def cct_14(arch, **kwargs):
    """cct_14"""
    return _cct(
        arch,
        num_layers=14,
        num_heads=6,
        mlp_ratio=3,
        embedding_dim=384,
        **kwargs)


def cct_2_3x2_32(
        img_size=32,
        positional_embedding='learnable',
        num_classes=10,
        **kwargs):
    """cct_2_3x2_32"""
    return cct_2(
        'cct_2_3x2_32',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_2_3x2_32_sine(
        img_size=32,
        positional_embedding='sine',
        num_classes=10,
        **kwargs):
    """cct_2_3x2_32_sine"""
    return cct_2(
        'cct_2_3x2_32_sine',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_4_3x2_32(
        img_size=32,
        positional_embedding='learnable',
        num_classes=10,
        **kwargs):
    """cct_2_3x2_32_sine"""
    return cct_4(
        'cct_4_3x2_32',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_4_3x2_32_sine(
        img_size=32,
        positional_embedding='sine',
        num_classes=10,
        **kwargs):
    """cct_2_3x2_32_sine"""
    return cct_4(
        'cct_4_3x2_32_sine',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_6_3x1_32(img_size=32, positional_embedding='learnable', num_classes=10,
                 **kwargs):
    """cct_2_3x2_32_sine"""
    return cct_6(
        'cct_6_3x1_32',
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_6_3x1_32_sine(
        img_size=32,
        positional_embedding='sine',
        num_classes=10,
        **kwargs):
    """cct_2_3x2_32_sine"""
    return cct_6(
        'cct_6_3x1_32_sine',
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_6_3x2_32(
        img_size=32,
        positional_embedding='learnable',
        num_classes=10,
        **kwargs):
    """cct_2_3x2_32_sine"""
    return cct_6(
        'cct_6_3x2_32',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_6_3x2_32_sine(
        img_size=32,
        positional_embedding='sine',
        num_classes=10,
        **kwargs):
    """cct_6_3x2_32_sine"""
    return cct_6(
        'cct_6_3x2_32_sine',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_3x1_32(
        img_size=32,
        positional_embedding='learnable',
        num_classes=10,
        **kwargs):
    """cct_7_3x1_32"""
    return cct_7(
        'cct_7_3x1_32',
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_3x1_32_sine(
        img_size=32,
        positional_embedding='sine',
        num_classes=10,
        **kwargs):
    """cct_7_3x1_32_sine"""
    return cct_7(
        'cct_7_3x1_32_sine',
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_3x1_32_c100(
        img_size=32,
        positional_embedding='learnable',
        num_classes=100,
        **kwargs):
    """cct_7_3x1_32_c100"""
    return cct_7(
        'cct_7_3x1_32_c100',
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_3x1_32_sine_c100(
        img_size=32,
        positional_embedding='sine',
        num_classes=100,
        **kwargs):
    """cct_7_3x1_32_sine_c100"""
    return cct_7(
        'cct_7_3x1_32_sine_c100',
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_3x2_32(
        img_size=32,
        positional_embedding='learnable',
        num_classes=10,
        **kwargs):
    """cct_7_3x2_32"""
    return cct_7(
        'cct_7_3x2_32',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_3x2_32_sine(
        img_size=32,
        positional_embedding='sine',
        num_classes=10,
        **kwargs):
    """cct_7_3x2_32_sine"""
    return cct_7(
        'cct_7_3x2_32_sine',
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_7_7x2_224(
        img_size=224,
        positional_embedding='learnable',
        num_classes=102):
    """cct_7_7x2_224"""
    return cct_7(
        'cct_7_7x2_224',
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes)


def cct_7_7x2_224_sine(
        img_size=224,
        positional_embedding='sine',
        num_classes=102,
        **kwargs):
    """cct_7_7x2_224_sine"""
    return cct_7(
        'cct_7_7x2_224_sine',
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_14_7x2_224(
        img_size=224,
        positional_embedding='learnable',
        num_classes=1000,
        **kwargs):
    """cct_14_7x2_224"""
    return cct_14(
        'cct_14_7x2_224',
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_14_7x2_384(
        img_size=384,
        positional_embedding='learnable',
        num_classes=1000,
        **kwargs):
    """cct_14_7x2_384"""
    return cct_14(
        'cct_14_7x2_384',
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)


def cct_14_7x2_384_fl(
        img_size=384,
        positional_embedding='learnable',
        num_classes=102,
        **kwargs):
    """cct_14_7x2_384_fl"""
    return cct_14(
        'cct_14_7x2_384_fl',
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs)
