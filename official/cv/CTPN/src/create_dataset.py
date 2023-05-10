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
from __future__ import division
import os
import numpy as np
from PIL import Image
from mindspore.mindrecord import FileWriter
from src.model_utils.config import config

def create_coco_label():
    """Create image label."""
    image_files = []
    image_anno_dict = {}
    coco_root = config.coco_root
    data_type = config.coco_train_data_type
    from src.coco_text import COCO_Text
    anno_json = config.cocotext_json
    ct = COCO_Text(anno_json)
    image_ids = ct.getImgIds(imgIds=ct.train,
                             catIds=[('legibility', 'legible')])
    for img_id in image_ids:
        image_info = ct.loadImgs(img_id)[0]
        file_name = image_info['file_name'][15:]
        anno_ids = ct.getAnnIds(imgIds=img_id)
        anno = ct.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        im = Image.open(image_path)
        width, _ = im.size
        for label in anno:
            bbox = label["bbox"]
            bbox_width = int(bbox[2])
            if 60 * bbox_width < width:
                continue
            x1, x2 = int(bbox[0]), int(bbox[0] + bbox[2])
            y1, y2 = int(bbox[1]), int(bbox[1] + bbox[3])
            annos.append([x1, y1, x2, y2] + [1])
        if annos:
            image_anno_dict[image_path] = np.array(annos)
            image_files.append(image_path)
    return image_files, image_anno_dict

def create_anno_dataset_label(train_img_dirs, train_txt_dirs):
    image_files = []
    image_anno_dict = {}
    # read
    img_basenames = []
    for file in os.listdir(train_img_dirs):
        # Filter git file.
        if 'gif' not in file:
            img_basenames.append(os.path.basename(file))
    img_names = []
    for item in img_basenames:
        temp1, _ = os.path.splitext(item)
        img_names.append((temp1, item))
    for img, img_basename in img_names:
        image_path = train_img_dirs + '/' + img_basename
        annos = []
        if len(img) == 6 and '_' not in img_basename:
            gt = open(train_txt_dirs + '/' + img + '.txt').read().splitlines()
            if img.isdigit() and int(img) > 1200:
                continue
            for img_each_label in gt:
                spt = img_each_label.replace(',', '').split(' ')
                if ' ' not in img_each_label:
                    spt = img_each_label.split(',')
                annos.append([spt[0], spt[1], str(int(spt[0]) + int(spt[2])), str(int(spt[1]) +  int(spt[3]))] + [1])
            if annos:
                image_anno_dict[image_path] = np.array(annos)
                image_files.append(image_path)
    return image_files, image_anno_dict

def create_icdar_svt_label(train_img_dir, train_txt_dir, prefix):
    image_files = []
    image_anno_dict = {}
    img_basenames = []
    for file_name in os.listdir(train_img_dir):
        if 'gif' not in file_name:
            img_basenames.append(os.path.basename(file_name))
    img_names = []
    for item in img_basenames:
        temp1, _ = os.path.splitext(item)
        img_names.append((temp1, item))
    for img, img_basename in img_names:
        image_path = train_img_dir + '/' + img_basename
        annos = []
        file_name = prefix + img + ".txt"
        file_path = os.path.join(train_txt_dir, file_name)
        gt = open(file_path, 'r', encoding='UTF-8-sig').read().splitlines()
        if not gt:
            annos.append([0, 0, 0, 0, 0])
        else:
            for img_each_label in gt:
                spt = img_each_label.replace(',', '').split(' ')
                if ' ' not in img_each_label:
                    spt = img_each_label.split(',')
                label = -1 if "#" in spt[-1] else 1
                annos.append([int(spt[0]), int(spt[1]), int(spt[2]), int(spt[3]), label])
        if annos:
            image_anno_dict[image_path] = np.array(annos)
            image_files.append(image_path)
    return image_files, image_anno_dict


def get_train_dataset_files_dicts(dataset, cfg):
    if dataset == "coco":
        return create_coco_label()
    if dataset == "flick":
        return create_anno_dataset_label(cfg.flick_train_path[0],
                                         cfg.flick_train_path[1])
    if dataset == "icdar11":
        return create_icdar_svt_label(cfg.icdar11_train_path[0],
                                      cfg.icdar11_train_path[1],
                                      cfg.icdar11_prefix)
    if dataset == "icdar13":
        return create_icdar_svt_label(cfg.icdar13_train_path[0],
                                      cfg.icdar13_train_path[1],
                                      cfg.icdar13_prefix)
    if dataset == "icdar15":
        return create_icdar_svt_label(cfg.icdar15_train_path[0],
                                      cfg.icdar15_train_path[1],
                                      cfg.icdar15_prefix)
    if dataset == "svt":
        return create_icdar_svt_label(cfg.svt_train_path[0], cfg.svt_train_path[1], "")
    if dataset == "icdar17_mlt":
        return create_icdar_svt_label(cfg.icdar17_mlt_train_path[0],
                                      cfg.icdar17_mlt_train_path[1],
                                      cfg.icdar17_mlt_prefix)
    raise ValueError(f"Do not support train dataset {dataset}")


def get_eval_dataset_files_dicts(dataset, cfg):
    if dataset == "icdar13":
        return create_icdar_svt_label(cfg.icdar13_test_path[0],
                                      cfg.icdar13_test_path[1],
                                      "")
    if dataset == "icdar15":
        return create_icdar_svt_label(cfg.icdar15_test_path[0],
                                      cfg.icdar15_test_path[1],
                                      cfg.icdar15_prefix)
    if dataset == "icdar17_mlt":
        return create_icdar_svt_label(cfg.icdar17_mlt_test_path[0],
                                      cfg.icdar17_mlt_test_path[1],
                                      cfg.icdar17_mlt_prefix)
    raise ValueError(f"Do not support eval dataset {dataset}")


def create_train_dataset(dataset_type):
    if dataset_type == "pretraining":
        # pretrianing: coco, flick, icdar2013 train, icdar2015, svt
        total_image_files = []
        total_image_anno_dict = {}
        for item in config.train_dataset:
            image_files, anno_dict = get_train_dataset_files_dicts(item, config)
            total_image_files += image_files
            total_image_anno_dict.update(anno_dict)
        data_to_mindrecord_byte_image(total_image_files, total_image_anno_dict, config.pretrain_dataset_path, \
            prefix="ctpn_pretrain.mindrecord", file_num=8)
    elif dataset_type == "finetune":
        # finetune: icdar2011, icdar2013 train, flick
        total_image_files = []
        total_image_anno_dict = {}
        for item in config.finetune_dataset:
            image_files, anno_dict = get_train_dataset_files_dicts(item, config)
            total_image_files += image_files
            total_image_anno_dict.update(anno_dict)
        data_to_mindrecord_byte_image(total_image_files, total_image_anno_dict, config.finetune_dataset_path, \
            prefix="ctpn_finetune.mindrecord", file_num=8)
    elif dataset_type == "test":
        # test: icdar2013 test
        total_image_files = []
        total_image_anno_dict = {}
        for item in config.test_dataset:
            image_files, anno_dict = get_eval_dataset_files_dicts(item, config)
            total_image_files += image_files
            total_image_anno_dict.update(anno_dict)
        data_to_mindrecord_byte_image(total_image_files, total_image_anno_dict, config.test_dataset_path, \
            prefix="ctpn_test.mindrecord", file_num=1)
    else:
        print("dataset_type should be pretraining, finetune, test")

def data_to_mindrecord_byte_image(image_files, image_anno_dict, dst_dir, prefix="cptn_mlt.mindrecord", file_num=1):
    """Create MindRecord file."""
    os.makedirs(dst_dir, exist_ok=True)
    mindrecord_path = os.path.join(dst_dir, prefix)
    snum = "" if file_num == 1 else "0"
    if os.path.exists(mindrecord_path + snum + ".db"):
        print(f"skip create {mindrecord_path}")
        return
    writer = FileWriter(mindrecord_path, file_num)

    ctpn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ctpn_json, "ctpn_json")
    image_files = sorted(image_files)
    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        print("img name is {}, anno is {}".format(image_name, annos))
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()

if __name__ == "__main__":
    create_train_dataset("pretraining")
    create_train_dataset("finetune")
    create_train_dataset("test")
