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
"""
python gendata.py
"""
import io
import os
import argparse
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


def gendata(read_path, save_dir):
    bins, _ = pickle.load(open(read_path, 'rb'), encoding='bytes')
    cnt = 0
    for bini in bins:
        data = plt.imread(io.BytesIO(bini), "jpg")
        cv2.imwrite(os.path.join(save_dir, str(cnt) + '_.jpg'), data)
        cv2.imwrite(os.path.join(save_dir, 'f_' + str(cnt) + '_.jpg'), np.fliplr(data))
        cnt += 1
        if cnt % 100 == 0:
            print('%d/%d' % (cnt, len(bins)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # Datasets
    parser.add_argument('--eval_url', default='./data/', type=str,
                        help='output path')
    parser.add_argument('--result_url', default='./data/', type=str)
    parser.add_argument('--target',
                        default='agedb_30,lfw,cfp_fp,calfw,cplfw',
                        help='test targets.')
    args = parser.parse_args()
    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        data_path = os.path.join(args.eval_url, name + ".bin")
        save_path = os.path.join(args.result_url, name)
        if os.path.exists(data_path):
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            gendata(data_path, save_path)
