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
import os
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/input/',
                        help="input data path")
    parser.add_argument('--eval_length', type=int, default=240,
                        help='eval length')
    parser.add_argument('--hop_size', type=int, default=256,
                        help='hop size')
    opts = parser.parse_args()
    data_path = opts.data_path
    eval_length = opts.eval_length
    hop_size = opts.hop_size

    file_list = os.listdir(data_path)
    data_list = []
    for data_name in file_list:
        if '.npy' in data_name:
            print(data_name)
            npypath = os.path.join(data_path, data_name)

            # data preprocessing
            meldata = np.load(npypath)
            meldata = (meldata + 5.0) / 5.0
            pad_node = 0

            if meldata.shape[1] < eval_length:
                pad_node = eval_length - meldata.shape[1]
                meldata = np.pad(meldata, ((0, 0), (0, pad_node)), mode='constant', constant_values=0.0)
            meldata_s = meldata[np.newaxis, :, 0:eval_length]
            new_data = meldata_s

            repeat_frame = eval_length // 8
            i = eval_length - repeat_frame
            length = eval_length

            while i < meldata.shape[1]:
                # data preprocessing
                meldata_s = meldata[:, i:i + length]
                if meldata_s.shape[1] != eval_length:
                    pad_node = hop_size * (eval_length - meldata_s.shape[1])
                    meldata_s = np.pad(meldata_s, ((0, 0), (0, eval_length - meldata_s.shape[1])), mode='edge')
                meldata_s = meldata_s[np.newaxis, :, :]
                new_data = np.concatenate((new_data, meldata_s), axis=1)
                i = i + length - repeat_frame
            out_file = npypath.replace('.npy', '_test.txt')
            np.savetxt(out_file, new_data.reshape((-1, eval_length)), fmt='%.18e')
            d = np.loadtxt(out_file, dtype=np.float32)
            data_list.append(data_name.replace('.npy', '_test.txt'))
            print((new_data.reshape((-1, eval_length))).shape)

    data_list_str = "\n".join(data_list)
    print(data_list_str)
    f = open(os.path.join(data_path, 'data_list.txt'), 'w')
    f.write(data_list_str)
    f.close()
