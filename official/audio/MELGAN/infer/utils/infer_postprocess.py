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
"""MelGAN eval"""
import argparse
import os

import numpy as np
from scipy.io.wavfile import write

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/input/',
                        help="input data path")
    parser.add_argument('--output_path', type=str, default='../mxbase/output',
                        help='output data path')
    parser.add_argument('--eval_length', type=int, default=240,
                        help='eval length')
    parser.add_argument('--hop_size', type=int, default=256,
                        help='hop size')
    parser.add_argument('--sample', type=int, default=22050,
                        help='sample')
    opts = parser.parse_args()
    data_path = opts.data_path
    output_path = opts.output_path
    eval_length = opts.eval_length
    hop_size = opts.hop_size
    sample = opts.sample

    data_list = os.listdir(output_path)
    print(data_list)
    for data_name in data_list:
        if 'test.txt' in data_name:
            txt_data = np.loadtxt(os.path.join(output_path, data_name), dtype=np.float32).reshape((-1, 61440))
            melname = data_name.replace('txt', 'npy').replace('restruction_', '').replace('_test', '')
            meldata = np.load(os.path.join(data_path, melname)).reshape((80, -1))

            pad_node = 0
            if meldata.shape[1] < eval_length:
                pad_node = eval_length - meldata.shape[1]

            # first frame
            wav_data = np.array([])
            output = txt_data[0].ravel()
            wav_data = np.concatenate((wav_data, output))

            # initialization parameters
            repeat_frame = eval_length // 8
            i = eval_length - repeat_frame
            length = eval_length
            num_weights = i
            interval = (hop_size * repeat_frame) // num_weights
            weights = np.linspace(0.0, 1.0, num_weights)

            while i < meldata.shape[1]:
                meldata_s = meldata[:, i:i + length]
                if meldata_s.shape[1] != eval_length:
                    pad_node = hop_size * (eval_length - meldata_s.shape[1])
                i = i + length - repeat_frame

            for idx in range(1, txt_data.shape[0]):
                # i-th frame
                output = txt_data[idx].ravel()
                lenwav = hop_size * repeat_frame
                lenout = 0
                # overlap
                for j in range(num_weights - 1):
                    wav_data[-lenwav:-lenwav + interval] = weights[-j - 1] * wav_data[-lenwav:-lenwav + interval] + \
                                                           weights[j] * output[lenout:lenout + interval]
                    lenwav = lenwav - interval
                    lenout = lenout + interval
                wav_data[-lenwav:] = weights[-num_weights] * wav_data[-lenwav:] + \
                                     weights[num_weights - 1] * output[lenout:lenout + lenwav]
                wav_data = np.concatenate((wav_data, output[hop_size * repeat_frame:]))
                i = i + length - repeat_frame

            if pad_node != 0:
                wav_data = wav_data[:-pad_node]

            # save as wav file
            wav_data = 32768.0 * wav_data
            out_path = os.path.join(output_path, 'restruction_' + data_name.replace('txt', 'wav'))
            write(out_path, sample, wav_data.astype('int16'))
