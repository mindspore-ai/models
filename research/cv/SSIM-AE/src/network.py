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

from mindspore import nn
from mindspore import ops


class AutoEncoder(nn.Cell):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()

        encoded_layers = []
        encoded_layers.extend(
            [
                nn.Conv2d(
                    cfg.input_channel,
                    cfg.flc,
                    4,
                    stride=2,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc,
                    4,
                    stride=2,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
            ]
        )
        if cfg.crop_size == 256:
            encoded_layers.extend(
                [
                    nn.Conv2d(
                        cfg.flc,
                        cfg.flc,
                        4,
                        stride=2,
                        weight_init="xavier_uniform",
                        has_bias=True,
                        pad_mode="pad",
                        padding=1,
                    ),
                    nn.LeakyReLU(alpha=0.2),
                ]
            )
        encoded_layers.extend(
            [
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc * 2,
                    4,
                    stride=2,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 4,
                    4,
                    stride=2,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc * 4,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc, cfg.z_dim, 8, stride=1, pad_mode="valid", weight_init="xavier_uniform", has_bias=True
                ),
            ]
        )
        self.encoded = nn.SequentialCell(encoded_layers)
        decoded_layers = []
        decoded_layers.extend(
            [
                nn.Conv2dTranspose(
                    cfg.z_dim, cfg.flc, 8, stride=1, pad_mode="valid", weight_init="xavier_uniform", has_bias=True
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 4,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2dTranspose(cfg.flc * 4, cfg.flc * 2, 4, stride=2, weight_init="xavier_uniform", has_bias=True),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2dTranspose(cfg.flc * 2, cfg.flc, 4, stride=2, weight_init="xavier_uniform", has_bias=True),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc,
                    3,
                    stride=1,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2dTranspose(cfg.flc, cfg.flc, 4, stride=2, weight_init="xavier_uniform", has_bias=True),
                nn.LeakyReLU(alpha=0.2),
            ]
        )
        if cfg.crop_size == 256:
            decoded_layers.extend(
                [
                    nn.Conv2dTranspose(cfg.flc, cfg.flc, 4, stride=2, weight_init="xavier_uniform", has_bias=True),
                    nn.LeakyReLU(alpha=0.2),
                ]
            )
        decoded_layers.extend(
            [
                nn.Conv2dTranspose(
                    cfg.flc,
                    cfg.input_channel,
                    4,
                    stride=2,
                    weight_init="xavier_uniform",
                    has_bias=True,
                    pad_mode="pad",
                    padding=1,
                ),
                nn.Sigmoid(),
            ]
        )
        self.decoded = nn.SequentialCell(decoded_layers)
        self.resize = nn.ResizeBilinear()

    def construct(self, input_batch):
        temp = self.encoded(input_batch)
        output_batch = self.decoded(temp)
        if (input_batch.shape[2] != output_batch.shape[2]) or (input_batch.shape[3] != output_batch.shape[3]):
            output_batch = self.resize(output_batch, input_batch.shape[2:])
        return output_batch


class SSIMLoss(nn.Cell):
    def __init__(self, max_val=1.0):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val
        self.loss_fn = nn.SSIM(max_val=self.max_val)
        self.reduce_mean = ops.ReduceMean()

    def construct(self, input_batch, target):
        output = self.loss_fn(input_batch, target)
        loss = 1 - self.reduce_mean(output)
        return loss


class NetWithLoss(nn.Cell):
    def __init__(self, net, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, input_batch):
        output = self._net(input_batch)
        return self._loss_fn(output, input_batch)
