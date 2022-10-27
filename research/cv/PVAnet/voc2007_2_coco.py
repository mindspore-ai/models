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
import json
import random
import shutil
import argparse
import xml.etree.ElementTree as et
import cv2
from tqdm import tqdm

voc_dict = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
            'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
            'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}


class COCOLoadAnnotation:
    def __init__(self):
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []
        self.category_set = voc_dict
        self.image_set = set()
        self.image_id = 0000000
        self.annotation_id = 0
        self.setCatItem()

    def setCatItem(self):
        for k, v in self.category_set.items():
            category_item = dict()
            category_item['supercategory'] = 'none'
            category_item['id'] = v
            category_item['name'] = k
            self.coco['categories'].append(category_item)
            self.category_set[k] = v

    def addImgItem(self, file_name, size):
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)

    def addAnnoItem(self, category_id, ann_dict):
        bbox = ann_dict['bbox']
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        seg.append(bbox[0])
        seg.append(bbox[1])
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        annotation_item['segmentation'].append(seg)
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = self.image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco['annotations'].append(annotation_item)


def write_xml_2_coco(coco_load_ann, dst='train2017'):
    if dst == 'train2017':
        images_dir = trainval_images_dir
        xmls_dir = trainval_xml_dir
    elif dst == 'val2017':
        images_dir = test_images_dir
        xmls_dir = test_xml_dir
    else:
        raise 'ERROR!'

    dst_root = os.path.join(args.voc_root, 'coco')
    dst_images_dir = os.path.join(dst_root, dst)
    dst_ann_dir = os.path.join(dst_root, 'annotations')
    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)
    if not os.path.exists(dst_ann_dir):
        os.makedirs(dst_ann_dir)

    json_name = 'instances_%s.json' % dst
    filename_list = os.listdir(xmls_dir)
    print('writing %s data...' % dst)
    random.shuffle(filename_list)
    size = dict()
    for filename in tqdm(filename_list):
        xml_file = os.path.join(xmls_dir, filename)
        img_path = os.path.join(images_dir, filename.split(".")[0] + '.jpg')
        img = cv2.imread(img_path)
        im_h, im_w, im_c = img.shape
        size['width'] = im_w
        size['height'] = im_h
        size['depth'] = im_c
        save_img_path = filename.split('.')[0] + '.jpg'
        coco_load_ann.addImgItem(save_img_path, size)

        tree = et.parse(xml_file)
        root = tree.getroot()
        for ob in root.findall("object"):
            bbox = ob.find("bndbox")
            class_name = ob.find("name").text
            object_name = class_name
            try:
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                w = int(xmax - xmin)
                h = int(ymax - ymin)
                box = [xmin, ymin, w, h]
            except ValueError:
                continue
            if not 0 <= xmin < xmax and 0 <= ymin < ymax:
                continue
            ann = {'bbox': box}
            category_id = coco_load_ann.category_set[object_name]
            coco_load_ann.addAnnoItem(category_id, ann)
        shutil.copy(img_path, dst_images_dir)
    json_file = os.path.join(dst_ann_dir, json_name)
    json.dump(coco_load_ann.coco, open(json_file, 'w'))
    print('%s data saved!' % dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_root", type=str, default=r'/mnt/data/savion/datasets/VOC2007')
    args = parser.parse_args()

    trainval_images_dir = os.path.join(args.voc_root, 'VOCtrainval/VOCdevkit/VOC2007/JPEGImages')
    trainval_xml_dir = os.path.join(args.voc_root, 'VOCtrainval/VOCdevkit/VOC2007/Annotations')
    test_images_dir = os.path.join(args.voc_root, 'VOCtest/VOCdevkit/VOC2007/JPEGImages')
    test_xml_dir = os.path.join(args.voc_root, 'VOCtest/VOCdevkit/VOC2007/Annotations')

    coco_load_ann_trainval = COCOLoadAnnotation()
    coco_load_ann_test = COCOLoadAnnotation()

    write_xml_2_coco(coco_load_ann_trainval, dst='train2017')
    write_xml_2_coco(coco_load_ann_test, dst='val2017')

    print('convert over')
