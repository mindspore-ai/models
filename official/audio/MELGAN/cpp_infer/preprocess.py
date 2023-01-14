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
"""310 data_processing"""
import os
import argparse
import numpy as np
from scipy.io.wavfile import write

parser = argparse.ArgumentParser(description='MelGAN')
parser.add_argument('--wav_path', type=str, default='', help='wav data path')
parser.add_argument('--bin_path', type=str, default='', help='bin data path')
parser.add_argument('--sample', type=int, default=22050, help='wav sample')
parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                    help='1 for wav to bin, 2 for bin to wav (Default: 1)')
args_opt = parser.parse_args()

if args_opt.mode == 1:
    path_all = args_opt.wav_path
    if not os.path.exists(args_opt.bin_path):
        os.mkdir(args_opt.bin_path)
else:
    path_all = args_opt.bin_path
    if not os.path.exists(args_opt.wav_path):
        os.mkdir(args_opt.wav_path)
filenames = os.listdir(path_all)

for filename in filenames:
    if args_opt.mode == 1:
        new_name = os.path.join(args_opt.bin_path, filename[:-4]+'.bin')
        temp = np.load(path_all+'/'+ filename)
        temp = (temp + 5) / 5
        if temp.shape[1] < 240:
            temp_1 = 240 - temp.shape[1]
            temp = np.pad(temp, ((0, 0), (0, temp_1)), mode='constant', constant_values=0.0)
        temp[:, :240].tofile(new_name)
    else:
        abc = np.fromfile(os.path.join(path_all, filename), dtype='float32')
        wav_data = 32768.0 * abc
        output_path = os.path.join(args_opt.wav_path, filename).replace('.bin', '.wav')
        write(output_path, args_opt.sample, wav_data.astype('int16'))
        print('get {}, please check it'.format(output_path))
