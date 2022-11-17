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
import json
import os
from collections import defaultdict
import cv2

parser = argparse.ArgumentParser(description='YOLOV4')
parser.add_argument('--data_url', type=str, default='./datasets', help='coco2017 datasets')
parser.add_argument('--train_url', type=str, default='./infer/data/models/', help='save txt file')
parser.add_argument('--val_url', type=str, default='./infer/data/images/', help='coco2017 val infer datasets')
args_opt, _ = parser.parse_known_args()

def name_box_parse(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
        annotations = data['annotations']
        for ant in annotations:
            image_id = ant['image_id']
            name = str("%012d.jpg" % image_id)
            cat = ant['category_id']

            if 1 <= cat <= 11:
                cat = cat - 1
            elif 13 <= cat <= 25:
                cat = cat - 2
            elif 27 <= cat <= 28:
                cat = cat - 3
            elif 31 <= cat <= 44:
                cat = cat - 5
            elif 46 <= cat <= 65:
                cat = cat - 6
            elif cat == 67:
                cat = cat - 7
            elif cat == 70:
                cat = cat - 9
            elif 72 <= cat <= 82:
                cat = cat - 10
            elif 84 <= cat <= 90:
                cat = cat - 11
            name_box_id[name].append([ant['bbox'], cat])


name_box_id = defaultdict(list)
id_name = dict()
name_box_parse(os.path.join(args_opt.data_url, 'annotations', 'instances_val2017.json'))

with open(os.path.join(args_opt.train_url, 'trainval.txt'), 'w') as g:
    ii = 0
    for idx, key in enumerate(name_box_id.keys()):
        print('trainval', key.split('/')[-1])

        g.write('%d ' % ii)
        ii += 1
        g.write(os.path.join(args_opt.val_url, key))

        print(os.path.join(args_opt.data_url, 'val2017', key))

        img = cv2.imread(os.path.join(args_opt.data_url, 'val2017', key))
        h, w, c = img.shape

        g.write(' %d %d' % (w, h))

        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d %d %d %d %d" % (
                int(info[1]), x_min, y_min, x_max, y_max
            )
            g.write(box_info)
        g.write('\n')
