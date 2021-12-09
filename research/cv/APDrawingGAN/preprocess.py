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
"""pre process for 310 inference"""

import os
from src.data import create_dataset
from src.data.single_dataloader import single_dataloader
from src.option.options_test import TestOptions

def preprocess_data():
    """ preprocess data """
    opt = TestOptions().get_settings()
    opt.rank = 0
    opt.group_size = 1

    dataset = create_dataset(opt)
    for data in dataset.create_dict_iterator(output_numpy=True):
        input_data = {}
        item = single_dataloader(data, opt)
        for d, v in item.items():
            if d == 'A_path':
                input_data[d] = v
            else:
                input_data[d] = v[0]
        filename = input_data['A_path'][0].split('/')[-1].split('.')[0]
        real_A = input_data['A']
        real_A_bg = input_data['bg_A']
        real_A_eyel = input_data['eyel_A']
        real_A_eyer = input_data['eyer_A']
        real_A_nose = input_data['nose_A']
        real_A_mouth = input_data['mouth_A']
        real_A_hair = input_data['hair_A']
        mask = input_data['mask']
        mask2 = input_data['mask2']
        real_A.tofile(os.path.join('./preprocess_Data/A', filename + '.bin'))
        real_A_bg.tofile(os.path.join('./preprocess_Data/bg_A', filename + '.bin'))
        real_A_eyel.tofile(os.path.join('./preprocess_Data/eyel_A', filename + '.bin'))
        real_A_eyer.tofile(os.path.join('./preprocess_Data/eyer_A', filename + '.bin'))
        real_A_nose.tofile(os.path.join('./preprocess_Data/nose_A', filename + '.bin'))
        real_A_mouth.tofile(os.path.join('./preprocess_Data/mouth_A', filename + '.bin'))
        real_A_hair.tofile(os.path.join('./preprocess_Data/hair_A', filename + '.bin'))
        mask.tofile(os.path.join('./preprocess_Data/mask', filename + '.bin'))
        mask2.tofile(os.path.join('./preprocess_Data/mask2', filename + '.bin'))

if __name__ == '__main__':
    preprocess_data()
