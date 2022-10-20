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

# This file was copied from project [ZhaoWeicheng][Pyramidbox.pytorch]

import os
from src.config import cfg

WIDER_ROOT = os.path.join(cfg.HOME, 'WIDERFACE')
train_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                               'wider_face_train_bbx_gt.txt')
val_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                             'wider_face_val_bbx_gt.txt')

WIDER_TRAIN = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
WIDER_VAL = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')


def parse_wider_file(root, file):
    with open(file, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in range(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


def wider_data_file():
    if not os.path.exists(cfg.FACE.FILE_DIR):
        os.mkdir(cfg.FACE.FILE_DIR)
    img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
    fw = open(cfg.FACE.TRAIN_FILE, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()

    img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
    fw = open(cfg.FACE.VAL_FILE, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()


if __name__ == '__main__':
    wider_data_file()
