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

from mindspore import  nn


class AutoEncoder(nn.Cell):
    def __init__(self, opt):
        super(AutoEncoder, self).__init__()

        encoded_layers = []
        encoded_layers.extend([
            nn.Conv2d(opt["input_channel"], opt["flc"], 4, stride=2,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"], opt["flc"], 4, stride=2,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2)
            ])
        if opt["data_augment"]["crop_size"] == 256:
            encoded_layers.extend([
                nn.Conv2d(opt["flc"], opt["flc"], 4, stride=2,
                          weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
                nn.LeakyReLU(alpha=0.2),
            ])
        encoded_layers.extend([
            nn.Conv2d(opt["flc"], opt["flc"], 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"], opt["flc"]*2, 4, stride=2,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"]*2, opt["flc"]*2, 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"] * 2, opt["flc"] * 4, 4, stride=2,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"] * 4, opt["flc"] * 2, 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"] * 2, opt["flc"], 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"], opt["z_dim"], 8, stride=1, pad_mode='valid',
                      weight_init="xavier_uniform", has_bias=True)
        ])
        self.encoded = nn.SequentialCell(encoded_layers)
        decoded_layers = []
        decoded_layers.extend([
            nn.Conv2dTranspose(opt["z_dim"], opt["flc"], 8, stride=1,
                               pad_mode='valid', weight_init="xavier_uniform", has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"], opt["flc"]*2, 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"]*2, opt["flc"]*4, 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2dTranspose(opt["flc"]*4, opt["flc"] * 2, 4, stride=2,
                               weight_init="xavier_uniform", has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"] * 2, opt["flc"] * 2, 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2dTranspose(opt["flc"]*2, opt["flc"], 4, stride=2,
                               weight_init="xavier_uniform", has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(opt["flc"], opt["flc"], 3, stride=1,
                      weight_init="xavier_uniform", has_bias=True, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2dTranspose(opt["flc"], opt["flc"], 4, stride=2,
                               weight_init="xavier_uniform", has_bias=True),
            nn.LeakyReLU(alpha=0.2)
        ])
        if opt["data_augment"]["crop_size"] == 256:
            decoded_layers.extend([
                nn.Conv2dTranspose(opt["flc"], opt["flc"], 4, stride=2,
                                   weight_init="xavier_uniform", has_bias=True),
                nn.LeakyReLU(alpha=0.2),
            ])
        decoded_layers.extend([
            nn.Conv2dTranspose(opt["flc"], opt["input_channel"], 4, stride=2,
                               weight_init="xavier_uniform",
                               has_bias=True, pad_mode='pad', padding=1),
            nn.Sigmoid(),
        ])
        self.decoded = nn.SequentialCell(decoded_layers)

    def construct(self, input_batch):
        temp = self.encoded(input_batch)
        output_batch = self.decoded(temp)
        return output_batch
