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
'''Preprocess for Relationnet MindX'''

import os
import random
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Preprocess for Relationnet MindX")
parser.add_argument("--sample_num_per_class", default=1)
parser.add_argument("--data_path", default='./data/dataset/omniglot_resized/')
parser.add_argument("--label_output_path", default='./data/label/')
parser.add_argument("--data_output_path", default="./data/input")
args = parser.parse_args()


a = os.listdir(args.data_path)

cnt = 0

finallist = []

for i in a:
    tmp = os.listdir(os.path.join(args.data_path, i))
    for j in tmp:
        cnt += 1
        finallist.append(os.path.join(args.data_path, i, j))

mean, std = np.array([0.92206, 0.08426], dtype=np.float32)

for i in range(1000):
    a = np.random.choice(a=cnt, size=5)
    sample_input = np.zeros((5, 1, 28, 28))
    test_input = np.zeros((5, 1, 28, 28))
    tmp_input = np.zeros((5, 1, 28, 28))
    test_label = np.array(range(5))
    Flip = random.choice([True, False])
    degree = random.choice([0, 90, 180, 270])
    for k, j in enumerate(a):
        image_name = os.listdir(finallist[j])
        image_num = len(image_name)
        a = np.random.choice(a=image_num, size=2)
        image_file1, image_file2 = image_name[a[0]], image_name[a[1]]
        sample_image = np.rot90(cv2.imread(finallist[j] + "/" + image_file1, cv2.IMREAD_GRAYSCALE), degree/90)
        test_image = np.rot90(cv2.imread(finallist[j] + "/" + image_file2, cv2.IMREAD_GRAYSCALE), degree/90)
        if Flip:
            sample_image = np.flip(sample_image, axis=1)
            test_image = np.flip(test_image, axis=1)
        sample_input[k] = sample_image
        tmp_input[k] = test_image
    np.random.shuffle(test_label)
    for m, n  in enumerate(test_label):
        test_input[m] = tmp_input[n]
    final = np.concatenate((sample_input, test_input), axis=0)
    final = final/np.array([255.0], dtype=np.float32)
    final = (final - mean) / std
    final = final.astype(np.float32)
    test_label = test_label.astype(np.int32)
    test_label.tofile(os.path.join(args.label_output_path, 'b' + str(i) + ".bin"))
    final.tofile(os.path.join(args.data_output_path, 'a' + str(i) + ".bin"))
