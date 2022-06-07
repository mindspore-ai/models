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

import os
import argparse
import cv2
import numpy as np
from PIL import Image

class TestDatasetGenerator:
    def __init__(self, test_data_dir):
        self.rgb_name_lists, self.depth_name_lists = self.get_rgb_and_depth_name_lists(test_data_dir)

    def __getitem__(self, index):
        rgb_name = self.rgb_name_lists[index]
        depth_name = self.depth_name_lists[index]
        rgb = Image.open(rgb_name)
        depth = Image.open(depth_name)
        rgb = np.array(rgb) / 255.0
        depth = np.array(depth) / 1000.0
        depth = depth.reshape((depth.shape[0], depth.shape[1], 1))
        return rgb, depth

    def __len__(self):
        return len(self.rgb_name_lists)

    def get_rgb_and_depth_name_lists(self, test_data_dir):
        rgb_name_lists = []
        depth_name_list = []
        file_lists = os.listdir(test_data_dir)
        file_lists.sort()
        for file_name in file_lists:
            if file_name[6:-4] == "colors":
                file_name = test_data_dir + "/" + file_name
                rgb_name_lists.append(file_name)
            if file_name[6:-4] == "depth":
                file_name = test_data_dir + "/" + file_name
                depth_name_list.append(file_name)

        return rgb_name_lists, depth_name_list


def preprocess(cfg):
    i = 0
    dataset = TestDatasetGenerator(cfg.data_path)
    rgb_list, depth_list = dataset.get_rgb_and_depth_name_lists(cfg.data_path)
    for rgb, depth in zip(rgb_list, depth_list):
        # the format of depth_png is single channel and 16 bits
        rgb_img = cv2.imread(rgb, cv2.IMREAD_COLOR)[:, :, ::-1]
        depth_png = cv2.imread(depth, cv2.CV_16UC1)
        rgb_img = rgb_img[12:468, 16:624]
        depth_png = depth_png[12:468, 16:624]
        rgb_final = cv2.resize(rgb_img, (304, 228), interpolation=cv2.INTER_LINEAR)
        depth_final = cv2.resize(depth_png, (74, 55), interpolation=cv2.INTER_LINEAR)
        rgb_final = np.float32(rgb_final)
        depth_final = np.float32(depth_final)
        rgb_final = np.transpose(rgb_final, axes=(2, 0, 1))
        # normalize and unit conversion
        rgb_final = rgb_final / 255.0
        depth_final = depth_final / 1000.0
        rgb_final = rgb_final.reshape((1, 3, 228, 304))
        depth_final = depth_final.reshape((1, 1, 55, 74))
        # save preprocess results
        rgb = rgb.replace("png", "bin")
        sub_path = cfg.preprocess_result_url
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        rgb_path = rgb.replace(cfg.data_path, cfg.preprocess_result_url)
        rgb_final.tofile(rgb_path)
        depth = depth.replace("png", "bin")
        depth_path = depth.replace(cfg.data_path, cfg.preprocess_result_url)
        depth_final.tofile(depth_path)
        i += 1
        if i % 50 == 0:
            print("Finish {} files".format(i))
    print("=" * 20, "export bin files finished", "=" * 20)


def parse_args():
    parser = argparse.ArgumentParser(description='DepthNet Inferring sdk')
    # Datasets
    parser.add_argument('--data_path', default='../input/data/nyu2_test', type=str,
                        help='test data path')
    parser.add_argument('--preprocess_result_url', default='../mxbase/mxbase_out/', type=str,
                        help='preprocess results path')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    preprocess(cfg=args)
    print("preprocessing done")
