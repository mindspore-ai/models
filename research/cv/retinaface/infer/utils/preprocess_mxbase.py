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

import argparse
import os
import numpy as np
import cv2

val_origin_size = True
val_save_result = True


def preprocess(args_opt):

    testset_folder = args_opt.input_path
    bin_folder = args_opt.output_path
    if not os.path.exists(bin_folder):
        os.mkdir(bin_folder)
    testset_label_path = args_opt.input_path + "val_label.txt"
    with open(testset_label_path, 'r') as f:
        _test_dataset = f.readlines()
        test_dataset = []
        for im_path in _test_dataset:
            if im_path.startswith('# '):
                test_dataset.append(im_path[2:-1])  # delete '# ...\n'

    if val_origin_size:
        h_max, w_max = 0, 0
        for img_name in test_dataset:
            image_path = os.path.join(testset_folder, img_name)
            _img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if _img.shape[0] > h_max:
                h_max = _img.shape[0]
            if _img.shape[1] > w_max:
                w_max = _img.shape[1]

        h_max = 5568
        w_max = 1056

    else:
        target_size = 1600
        max_size = 2176

    # preprocessing begin
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(testset_folder, img_name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        if val_origin_size:
            assert img.shape[0] <= h_max and img.shape[1] <= w_max
            image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
            image_t[:, :] = (104.0, 117.0, 123.0)
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t
        else:
            im_size_min = np.min(img.shape[0:2])
            im_size_max = np.max(img.shape[0:2])
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)

            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            assert img.shape[0] <= max_size and img.shape[1] <= max_size
            image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
            image_t[:, :] = (104.0, 117.0, 123.0)
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        # save bin file
        flag = img_name.find("/")
        sub_path = img_name[:flag]
        sub_path = os.path.join(bin_folder, sub_path)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        img_name = img_name.replace("jpg", "bin")
        file_path = os.path.join(bin_folder, img_name)
        img.tofile(file_path)
        if i % 50 == 0:
            print("Finish {} files".format(i))
    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mxbase Inferring preprocess')
    # Datasets
    parser.add_argument('--input_path', default='../data/input/val_images/', type=str,
                        help='original image path')
    parser.add_argument('--output_path', default='../data/input/bin_files/', type=str,
                        help='preprocessed bin files save path')
    args = parser.parse_args()

    preprocess(args_opt=args)
    print("preprocessing done")
