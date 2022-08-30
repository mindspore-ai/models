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
import json
from xml.etree import ElementTree  as ET
from tqdm import tqdm
cls_names = ['with_mask', 'without_mask', 'mask_worn_incorrectly']
cls_map = {}
for i, n in enumerate(cls_names):
    cls_map[n] = i + 1
def  _parser_args():
    """parse input"""
    parser = argparse.ArgumentParser("convert xml to coco")
    parser.add_argument('--data_path', type=str, required=True, help='the path of data')
    parser.add_argument('--save_path', type=str, required=True, help='save path')
    args_, _ = parser.parse_known_args()
    return args_

def xml2coco(data_path, save_path):
    xml_list = os.listdir(os.path.join(data_path, 'annotations/'))
    coco = dict()
    coco['images'] = []
    coco['annotations'] = []
    coco['categories'] = []
    for cls in cls_map:
        cat = dict()
        cat['name'] = cls
        cat['id'] = cls_map[cls]
        cat['supercategory'] = cls
        coco['categories'].append(cat)
    i_id = 0
    a_id = 0
    for xml in tqdm(xml_list):
        tree = ET.parse(os.path.join(data_path, 'annotations/', xml))
        image_name = tree.find('./filename').text
        img_h = int(tree.find('./size/height').text)
        img_w = int(tree.find('./size/width').text)

        objs = tree.findall('./object')
        image = dict()
        image['file_name'] = image_name
        image['width'] = img_w
        image['height'] = img_h
        image['id'] = i_id
        coco['images'].append(image)
        for obj in objs:
            name = obj.find('./name').text
            x_min = int(obj.find('./bndbox/xmin').text)
            x_max = int(obj.find('./bndbox/xmax').text)
            y_min = int(obj.find('./bndbox/ymin').text)
            y_max = int(obj.find('./bndbox/ymax').text)
            if x_min <= 0 or y_min <= 0 or x_max - x_min <= 0 or y_max - y_min <= 0:
                continue
            annotation = dict()
            annotation['id'] = a_id
            annotation['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
            annotation['area'] = (x_max - x_min) * (y_max - y_min)
            annotation['segmentation'] = [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]
            if name not in cls_names:
                continue
            cat_id = cls_map[name]
            annotation['category_id'] = cat_id
            annotation['image_id'] = i_id
            annotation['iscrowd'] = 0
            a_id += 1
            coco['annotations'].append(annotation)
            a_id += 1
        i_id += 1
    with open(save_path, 'w') as f:
        json.dump(coco, f, indent='\t')


if __name__ == '__main__':
    args = _parser_args()
    xml2coco(args.data_path, args.save_path)
