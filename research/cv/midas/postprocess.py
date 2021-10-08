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
"""preprocess"""
import os
import glob
import json
import argparse
import cv2
from PIL import Image
import numpy as np
from src.util import depth_read_kitti, depth_read_sintel, BadPixelMetric




def get_eval_result(img_path, result_path, data_name):
    """
    eval result bin format to numpy format
    """
    all_path = img_path.split('/')
    tempfilename = all_path[-1]
    filename, _ = os.path.splitext(tempfilename)
    id_feature_result_file = ""
    if data_name == "Kitti":
        seq = all_path[-3]
        id_feature_result_file = os.path.join(result_path, seq, filename + "_0.bin")

    elif data_name == "Sintel":
        seq = all_path[-2]
        id_feature_result_file = os.path.join(result_path, seq, filename + "_0.bin")

    elif data_name == "TUM":
        id_feature_result_file = os.path.join(result_path, filename + "_0.bin")
    else:
        print("no data_name")

    id_feature = np.fromfile(id_feature_result_file, dtype=np.float32).reshape(1, 384, 384)

    return id_feature


def eval_Kitti(data_path, result_path):
    """
    eval Kitti dataset
    """
    image_path = glob.glob(os.path.join(data_path, '*', 'image', '*.png'))
    print("image_path is ", image_path)
    loss_sum = 0
    metric = BadPixelMetric(1.25, 80, 'KITTI')
    for ind, file_name in enumerate(image_path):
        print(f"processing: {ind + 1} / {len(image_path)}")
        prediction = get_eval_result(file_name, result_path, "Kitti")
        all_path = file_name.split('/')
        depth_path_name = all_path[-1].split('.')[0]
        depth = depth_read_kitti(os.path.join(data_path, all_path[-3], 'depth', depth_path_name + '.png'))  # (436,1024)
        depth = np.expand_dims(depth, 0)
        mask = (depth > 0) & (depth < 80)

        prediction = np.squeeze(prediction)
        prediction = cv2.resize(prediction, (mask.shape[2], mask.shape[1]))

        loss = metric(prediction, depth, mask)
        loss_sum += loss

    print(f"Kitti bad pixel: {loss_sum / len(image_path):.3f}")
    return loss_sum / len(image_path)

def eval_TUM(data_path, result_path):
    """
    eval TUM dataset
    """
    metric = BadPixelMetric(1.25, 10, 'TUM')
    loss_sum = 0
    file_path = os.path.join(data_path, 'rgbd_dataset_freiburg2_desk_with_person', 'associate.txt')
    num = 0
    all_path = file_path.split('/')
    for line in open(file_path):
        num += 1
        print(f"processing: {num}")
        data = line.split('\n')[0].split(' ')
        image_path = os.path.join(data_path, all_path[-2], data[0])  # (480,640,3)
        depth_path = os.path.join(data_path, all_path[-2], data[1])  # (480,640,3)
        prediction = get_eval_result(image_path, result_path, "TUM")
        depth = cv2.imread(depth_path)[:, :, 0] / 5000
        depth = np.expand_dims(depth, 0)
        mask = (depth > 0) & (depth < 10)

        prediction = np.squeeze(prediction)
        prediction = cv2.resize(prediction, (mask.shape[2], mask.shape[1]))
        loss = metric(prediction, depth, mask)
        loss_sum += loss
    print(f"TUM bad pixel: {loss_sum / num:.3f}")
    return loss_sum / num

def eval_Sintel(data_path, result_path):
    """
    eval Sintel dataset
    """
    image_path = glob.glob(os.path.join(data_path, 'final_left', '*', '*.png'))
    loss_sum = 0
    metric = BadPixelMetric(1.25, 72, 'sintel')
    for ind, file_name in enumerate(image_path):
        print(f'processing: {ind + 1} / {len(image_path)}')
        prediction = get_eval_result(file_name, result_path, "Sintel")
        all_path = file_name.split('/')
        depth_path_name = all_path[-1].split('.')[0]
        depth = depth_read_sintel(os.path.join(data_path, 'depth', all_path[-2], depth_path_name + '.dpt'))

        mask1 = np.array(Image.open(os.path.join(data_path, 'occlusions', all_path[-2], all_path[-1]))).astype(int)
        mask1 = mask1 / 255

        mask = (mask1 == 1)&(depth > 0) & (depth < 80)
        depth = np.expand_dims(depth, 0)
        mask = np.expand_dims(mask, 0)
        prediction = np.squeeze(prediction)
        prediction = cv2.resize(prediction, (mask.shape[2], mask.shape[1]))

        loss = metric(prediction, depth, mask)
        print('loss is ', loss)
        loss_sum += loss

    print(f"Sintel bad pixel: {loss_sum / len(image_path):.3f}")
    return loss_sum / len(image_path)


def run_eval(config):
    """
    run eval
    """
    results = {}
    if config.dataset_name == 'Kitti':
        data_path = config.dataset_path + '/Kitti_raw_data/'
        result_path = config.result_path + '/Kitti/'
        results[config.dataset_name] = eval_Kitti(data_path=data_path, result_path=result_path)
    elif config.dataset_name == 'Sintel':
        data_path = config.dataset_path + '/Sintel/'
        result_path = config.result_path + '/Sintel/'
        results[config.dataset_name] = eval_Sintel(data_path=data_path, result_path=result_path)
    elif config.dataset_name == 'TUM':
        data_path = config.dataset_path + '/TUM/'
        result_path = config.result_path + '/TUM/TUM/'
        results[config.dataset_name] = eval_TUM(data_path=data_path, result_path=result_path)
    else:
        print("dataset error ")
    print(results)
    json.dump(results, open('./result_val.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../dataset/',
                        help='dataset path')
    parser.add_argument('--result_path', default='./result_Files',
                        help='bin path')
    parser.add_argument('--dataset_name', default='TUM',
                        help='dataset name')
    args = parser.parse_args()
    run_eval(args)
