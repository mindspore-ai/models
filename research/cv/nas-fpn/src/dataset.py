# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""retinanet dataset"""

from __future__ import division

import os
import numpy as np
import cv2
import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter
from src.model_utils.config import config
from src.box_utils import jaccard_numpy, retinanet_bboxes_encode

def _rand(a=0., b=1.):
    """Generate random."""
    return np.random.rand() * (b - a) + a

def get_imageId_from_fileName(filename):
    """Get imageID from fileName"""
    filename = os.path.splitext(filename)[0]
    if filename.isdigit():
        return int(filename)
    return id_iter

def random_sample_crop(image, boxes):
    """Random Crop the image and boxes"""
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])

    if min_iou is None:
        return image, boxes

    # max trails (50)
    for _ in range(50):
        image_t = image

        w = _rand(0.3, 1.0) * width
        h = _rand(0.3, 1.0) * height

        # aspect ratio constraint b/t .5 & 2
        if h / w < 0.5 or h / w > 2:
            continue

        left = _rand() * (width - w)
        top = _rand() * (height - h)

        rect = np.array([int(top), int(left), int(top + h), int(left + w)])
        overlap = jaccard_numpy(boxes, rect)

        # dropout some boxes
        drop_mask = overlap > 0
        if not drop_mask.any():
            continue

        if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
            continue

        image_t = image_t[rect[0]:rect[2], rect[1]:rect[3], :]

        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        # mask in that both m1 and m2 are true
        mask = m1 * m2 * drop_mask

        # have any valid boxes? try again if not
        if not mask.any():
            continue

        # take only matching gt boxes
        boxes_t = boxes[mask, :].copy()

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
        boxes_t[:, :2] -= rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
        boxes_t[:, 2:4] -= rect[:2]

        return image_t, boxes_t
    return image, boxes


def preprocess_fn(img_id, image, box, is_training):
    """Preprocess function for dataset."""
    cv2.setNumThreads(2)
    def _infer_data(image, input_shape):
        img_h, img_w, _ = image.shape
        input_h, input_w = input_shape

        image = cv2.resize(image, (input_w, input_h))

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        return img_id, image, np.array((img_h, img_w), np.float32)

    def _data_aug(image, box, is_training, image_size=(600, 600)):
        """Data augmentation function."""
        ih, iw, _ = image.shape
        w, h = image_size

        if not is_training:
            return _infer_data(image, image_size)

        # Random crop
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw, _ = image.shape

        # Resize image
        image = cv2.resize(image, (w, h))

        # Flip image or not
        flip = _rand() < .5
        if flip:
            image = cv2.flip(image, 1, dst=None)

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        box[:, [0, 2]] = box[:, [0, 2]] / ih
        box[:, [1, 3]] = box[:, [1, 3]] / iw

        if flip:
            box[:, [1, 3]] = 1 - box[:, [3, 1]]

        box, label, num_match = retinanet_bboxes_encode(box)
        return image, box, label, num_match

    return _data_aug(image, box, is_training, image_size=config.img_shape)

def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = config.coco_root
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type

    # Classes need to train or test.
    train_cls = config.coco_classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instances_set.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    images = []
    image_path_dict = {}
    image_anno_dict = {}

    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        iscrowd = False
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            iscrowd = iscrowd or label["iscrowd"]
            if class_name in train_cls:
                x_min, x_max = bbox[0], bbox[0] + bbox[2]
                y_min, y_max = bbox[1], bbox[1] + bbox[3]
                annos.append(list(map(round, [y_min, x_min, y_max, x_max])) + [train_cls_dict[class_name]])

        if not is_training and iscrowd:
            continue
        if len(annos) >= 1:
            images.append(img_id)
            image_path_dict[img_id] = image_path
            image_anno_dict[img_id] = np.array(annos)

    return images, image_path_dict, image_anno_dict


def anno_parser(annos_str):
    """Parse annotation from string to list."""
    annos = []
    for anno_str in annos_str:
        anno = list(map(int, anno_str.strip().split(',')))
        annos.append(anno)
    return annos


def filter_valid_data(image_dir, anno_path):
    """Filter valid image file, which both in image_dir and anno_path."""
    images = []
    image_path_dict = {}
    image_anno_dict = {}
    if not os.path.isdir(image_dir):
        raise RuntimeError("Path given is not valid.")
    if not os.path.isfile(anno_path):
        raise RuntimeError("Annotation file is not valid.")

    with open(anno_path, "rb") as f:
        lines = f.readlines()
    for img_id, line in enumerate(lines):
        line_str = line.decode("utf-8").strip()
        line_split = str(line_str).split(' ')
        file_name = line_split[0]
        image_path = os.path.join(image_dir, file_name)
        if os.path.isfile(image_path):
            images.append(img_id)
            image_path_dict[img_id] = image_path
            image_anno_dict[img_id] = anno_parser(line_split[1:])

    return images, image_path_dict, image_anno_dict

def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="retinanet.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        images, image_path_dict, image_anno_dict = create_coco_label(is_training)
    else:
        images, image_path_dict, image_anno_dict = filter_valid_data(config.image_dir, config.anno_path)

    retinanet_json = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(retinanet_json, "retinanet_json")

    for img_id in images:
        image_path = image_path_dict[img_id]
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[img_id], dtype=np.int32)
        img_id = np.array([img_id], dtype=np.int32)
        row = {"img_id": img_id, "image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_retinanet_dataset(mindrecord_file, batch_size, repeat_num, device_num=1, rank=0,
                             is_training=True, num_parallel_workers=24):
    """Creatr retinanet dataset with MindDataset."""
    ds = de.MindDataset(mindrecord_file, columns_list=["img_id", "image", "annotation"], num_shards=device_num,
                        shard_id=rank, num_parallel_workers=num_parallel_workers, shuffle=is_training)
    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    change_swap_op = C.HWC2CHW()
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    color_adjust_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    compose_map_func = (lambda img_id, image, annotation: preprocess_fn(img_id, image, annotation, is_training))
    if is_training:
        output_columns = ["image", "box", "label", "num_match"]
        trans = [color_adjust_op, normalize_op, change_swap_op]
    else:
        output_columns = ["img_id", "image", "image_shape"]
        trans = [normalize_op, change_swap_op]
    ds = ds.map(operations=compose_map_func, input_columns=["img_id", "image", "annotation"],
                output_columns=output_columns, column_order=output_columns,
                python_multiprocessing=is_training,
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=trans, input_columns=["image"], python_multiprocessing=is_training,
                num_parallel_workers=num_parallel_workers)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_num)
    return ds

def create_mindrecord(dataset="coco", prefix="retinanet.mindrecord", is_training=True):
    """create mindrecord dataset"""
    print("Start create dataset!")
    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is retinanet.mindrecord0, 1, ... file_num.

    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", is_training, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
    return mindrecord_file
