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
import numpy as np

parser = argparse.ArgumentParser(description="Preprocess WIDER Face Annotation file")
parser.add_argument('-f', type=str, default='dataset/wider_face_train_bbx_gt.txt',
                    help="Original wider face train annotation file")
args = parser.parse_args()

wider_face_train = open('dataset/wider_face_train.txt', 'w')

with open(args.f, 'r') as f:
    lines = f.readlines()
    total_num = len(lines)

    i = 0
    while i < total_num:
        image_name = lines[i].strip().rstrip('.jpg')
        wider_face_train.write(image_name + ' ')
        face_num = int(lines[i+1].strip())
        if face_num == 0:
            for _ in range(4):
                wider_face_train.write(str(0.0) + ' ')
            wider_face_train.write('\n')
            i = i + 3
            continue
        box_list = []
        for j in range(face_num):
            box = lines[i+2+j].split(' ')
            x = float(box[0])
            y = float(box[1])
            w = float(box[2])
            h = float(box[3])
            box_list.append([x, y, x + w, y + h])
        box_list = np.array(box_list).flatten()
        for num in box_list:
            wider_face_train.write(str(num) + ' ')
        wider_face_train.write('\n')
        i = i + face_num + 2

wider_face_train.close()
print("wider_face_train.txt has been successfully created in dataset dir!!")
