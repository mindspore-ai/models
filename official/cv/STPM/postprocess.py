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
"""postprocess"""
import os
import argparse
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from src.dataset import createDataset

parser = argparse.ArgumentParser(description='postprocess')

parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--label_dir', type=str, default='')
parser.add_argument('--category', type=str, default='screw')
parser.add_argument('--out_size', type=int, default=256, help="out size")
args = parser.parse_args()


def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    if norm.all() == 0:
        return v
    return v / norm


def cal_anomaly_map(fs_list, ft_list, out_size=224):
    """cal_anomaly_map"""
    anomaly_map = np.ones([out_size, out_size])
    for j in range(len(ft_list)):
        fs = fs_list[j]
        ft = ft_list[j]
        fs_norm = normalize(fs)
        ft_norm = normalize(ft)
        num = fs_norm * ft_norm
        cos = np.sum(num, 1)
        a_map = 1 - cos
        a_map = np.expand_dims(a_map, 1)
        a_map = a_map[0, 0, :, :]
        a_map = cv2.resize(a_map, (out_size, out_size))
        anomaly_map *= a_map
    return anomaly_map


if __name__ == '__main__':
    _, ds_test = createDataset(args.data_dir, args.category)

    gt_list_px_lvl = []
    pred_list_px_lvl = []
    gt_list_img_lvl = []
    pred_list_img_lvl = []

    for i, data in enumerate(ds_test.create_dict_iterator()):
        gt = data['gt']
        label = data['label']

        file_name_0 = os.path.join(args.result_dir, 'data_img_' + str(i) + '_0.bin')
        file_name_1 = os.path.join(args.result_dir, 'data_img_' + str(i) + '_1.bin')
        file_name_2 = os.path.join(args.result_dir, 'data_img_' + str(i) + '_2.bin')
        file_name_3 = os.path.join(args.result_dir, 'data_img_' + str(i) + '_3.bin')
        file_name_4 = os.path.join(args.result_dir, 'data_img_' + str(i) + '_4.bin')
        file_name_5 = os.path.join(args.result_dir, 'data_img_' + str(i) + '_5.bin')

        features_s = [np.fromfile(file_name_0, np.float32).reshape(1, 64, 64, 64),
                      np.fromfile(file_name_1, np.float32).reshape(1, 128, 32, 32),
                      np.fromfile(file_name_2, np.float32).reshape(1, 256, 16, 16)]
        features_t = [np.fromfile(file_name_3, np.float32).reshape(1, 64, 64, 64),
                      np.fromfile(file_name_4, np.float32).reshape(1, 128, 32, 32),
                      np.fromfile(file_name_5, np.float32).reshape(1, 256, 16, 16)]
        A_map = cal_anomaly_map(features_s, features_t, out_size=args.out_size)
        gt_np = gt.asnumpy()[0, 0].astype(int)

        gt_list_px_lvl.extend(gt_np.ravel())
        pred_list_px_lvl.extend(A_map.ravel())
        gt_list_img_lvl.append(label.asnumpy()[0])
        pred_list_img_lvl.append(A_map.max())

    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

    print("category: ", args.category)
    print("Total pixel-level auc-roc score : ", pixel_auc)
    print("Total image-level auc-roc score :", img_auc)
