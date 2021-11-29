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
"""test"""
import os
import glob
import argparse as arg
import numpy as np
from PIL import Image



if __name__ == '__main__':

    parser = arg.ArgumentParser(description='MxBase Infer data postprocess')
    parser.add_argument('--data_url', required=False, default='../result/', help='Location of bin data')
    parser.add_argument('--result_url', required=False, default='../result_png/', help='Location of result data')
    args = parser.parse_args()

    file_list = glob.glob(args.data_url + "*")
    print(file_list)
    for file in file_list:
        data = np.fromfile(file, '<f4')
        print(data.shape)
        data = np.reshape(data, (1, 3, 2848, 4256))
        data = np.transpose(np.squeeze(data, 0), (1, 2, 0))
        data = np.minimum(np.maximum(data, 0), 1)
        data = np.trunc(data * 255)
        data = data.astype(np.int8)
        image = Image.fromarray(data, 'RGB')
        file_name = os.path.basename(file)
        image.save(args.result_url+file_name[:-4] + '.png')
