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
"""MelGAN eval"""
import os
import numpy as np
from scipy.io.wavfile import write

from mindspore import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.tensor import Tensor
import mindspore.context as context

from src.model import Generator
from src.model_utils.config import config as cfg

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


if __name__ == '__main__':
    context.set_context(device_id=cfg.device_id)
    if not os.path.exists(cfg.output_path):
        os.mkdir(cfg.output_path)

    net_G = Generator(alpha=cfg.leaky_alpha)
    net_G.set_train(False)

    # load checkpoint
    param_dict = load_checkpoint(cfg.eval_model_path)
    load_param_into_net(net_G, param_dict)
    print('load model done !')

    model = Model(net_G)

    # get list
    mel_path = cfg.eval_data_path
    data_list = os.listdir(mel_path)

    for data_name in data_list:

        melpath = os.path.join(mel_path, data_name)

        # data preprocessing
        meldata = np.load(melpath)
        meldata = (meldata + 5.0) / 5.0
        pad_node = 0
        if meldata.shape[1] < cfg.eval_length:
            pad_node = cfg.eval_length - meldata.shape[1]
            meldata = np.pad(meldata, ((0, 0), (0, pad_node)), mode='constant', constant_values=0.0)
        meldata_s = meldata[np.newaxis, :, 0:cfg.eval_length]

        # first frame
        wav_data = np.array([])
        output = model.predict(Tensor(meldata_s)).asnumpy().ravel()
        wav_data = np.concatenate((wav_data, output))

        # initialization parameters
        repeat_frame = cfg.eval_length // 8
        i = cfg.eval_length - repeat_frame
        length = cfg.eval_length
        num_weights = i
        interval = (cfg.hop_size*repeat_frame) // num_weights
        weights = np.linspace(0.0, 1.0, num_weights)

        while i < meldata.shape[1]:
            # data preprocessing
            meldata_s = meldata[:, i:i+length]
            if meldata_s.shape[1] != cfg.eval_length:
                pad_node = cfg.hop_size * (cfg.eval_length-meldata_s.shape[1])
                meldata_s = np.pad(meldata_s, ((0, 0), (0, cfg.eval_length-meldata_s.shape[1])), mode='edge')
            meldata_s = meldata_s[np.newaxis, :, :]

            # i-th frame
            output = model.predict(Tensor(meldata_s)).asnumpy().ravel()
            print('output{}={}'.format(i, output))
            lenwav = cfg.hop_size*repeat_frame
            lenout = 0

            # overlap
            for j in range(num_weights-1):
                wav_data[-lenwav:-lenwav+interval] = weights[-j-1] * wav_data[-lenwav:-lenwav+interval] +\
                                                     weights[j] * output[lenout:lenout+interval]
                lenwav = lenwav - interval
                lenout = lenout + interval
            wav_data[-lenwav:] = weights[-num_weights] * wav_data[-lenwav:] +\
                                 weights[num_weights-1] * output[lenout:lenout+lenwav]
            wav_data = np.concatenate((wav_data, output[cfg.hop_size*repeat_frame:]))
            i = i + length - repeat_frame

        if pad_node != 0:
            wav_data = wav_data[:-pad_node]

        # save as wav file
        wav_data = 32768.0 * wav_data
        out_path = os.path.join(cfg.output_path, 'restruction_' + data_name.replace('npy', 'wav'))
        write(out_path, cfg.sample, wav_data.astype('int16'))

        print('{} done!'.format(data_name))
