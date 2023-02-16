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
""" SGAE model """
from collections import OrderedDict
from  ast import literal_eval
from mindspore import nn

class SGAE(nn.Cell):
    ''' SGAE model '''
    def __init__(self, input_shape, hidden_dim_input, layer_list=None):
        super(SGAE, self).__init__()
        layer_list = layer_list or [20, 40, 80, 256, 1024]
        encoder_dict = OrderedDict()
        decoder_dict = OrderedDict()
        if hidden_dim_input == 'auto':
            hidden_dim = []
            if input_shape < 20:
                hidden_dim = [10, input_shape]
            else:
                for idx in range(len(layer_list)):
                    if layer_list[idx] < input_shape:
                        hidden_dim.append(layer_list[idx])
                    elif input_shape - layer_list[idx-1] < layer_list[idx-1] * 0.1:
                        hidden_dim[-1] = input_shape
                    else:
                        hidden_dim.append(input_shape)
                if input_shape > layer_list[-1] * 1.5:
                    hidden_dim.append(input_shape)
                else:
                    hidden_dim[-1] = input_shape
            hidden_dim.reverse()
            print("hidden_dim", hidden_dim)
        else:
            hidden_dim_input = literal_eval(hidden_dim_input)
            hidden_dim = hidden_dim_input[:]
            hidden_dim.insert(0, input_shape)
        # encoder
        for idx in range(len(hidden_dim) - 1):
            encoder_dict[f'en_lr{idx+1}'] = nn.Dense(
                hidden_dim[idx], hidden_dim[idx+1])
            encoder_dict[f'en_relu{idx+1}'] = nn.ReLU()
            # self.encoder_dict[f'en_tanh{idx+1}']=nn.Tanh()
        # decoder
        for idx in range(len(hidden_dim) - 1, 1, -1):
            decoder_dict[f'de_lr{idx}'] = nn.Dense(
                hidden_dim[idx], hidden_dim[idx-1])
            decoder_dict[f'de_relu{idx}'] = nn.ReLU()
            # self.encoder_dict[f'de_tanh{idx+1}']=nn.Tanh()

        decoder_dict[f'de_lr{1}'] = nn.Dense(hidden_dim[1], hidden_dim[0])

        self.encoder = nn.SequentialCell(encoder_dict)
        self.decoder = nn.SequentialCell(decoder_dict)

        # scoring network
        scores_dict = OrderedDict()
        scores_dict[f'scores1'] = nn.Dense(hidden_dim[-1], 10)
        scores_dict[f'scores2'] = nn.Dense(10, 1)
        self.scores = nn.SequentialCell(scores_dict)

    def construct(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        scores = self.scores(enc)
        return scores, dec, enc
        