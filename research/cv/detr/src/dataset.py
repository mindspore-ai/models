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
"""dataset"""
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from mindspore import dataset as ds
from pycocotools.coco import COCO

import src.transforms as src_transforms


class CocoDataset:
    """coco dataset"""
    def __init__(self, root, anno_file, num_queries=100, transforms=None, num_classes=91):
        self.root = root
        self.anno_file = anno_file
        self.transforms = transforms
        self.coco = COCO(anno_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_queries = num_queries
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))[..., ::-1]  # BGR to RGB
        target = {'image_id': img_id, 'annotations': target}
        img, target = prepare_data(img, target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        bboxes = target['boxes']
        labels = target['labels']
        orig_size = target['orig_size']
        img_id = target['image_id']
        n_boxes = bboxes.shape[0]
        if n_boxes < self.num_queries:
            bboxes_padded = np.zeros((self.num_queries, 4), dtype=bboxes.dtype) + 1e6
            labels_padded = np.zeros((self.num_queries,), dtype=labels.dtype) + self.num_classes
            bboxes_padded[:n_boxes] = bboxes
            labels_padded[:n_boxes] = labels
            return img, bboxes_padded, labels_padded, orig_size, n_boxes, img_id
        return img, bboxes, labels, orig_size, n_boxes, img_id


def prepare_data(image, target):
    """prepare data"""
    h, w, _ = image.shape

    image_id = target["image_id"]
    image_id = np.array(image_id)

    anno = target["annotations"]

    anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)

    classes = [obj["category_id"] for obj in anno]
    classes = np.array(classes, dtype=np.int64)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["image_id"] = image_id

    # for conversion to coco api
    area = np.array([obj["area"] for obj in anno])
    iscrowd = np.array([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    target["orig_size"] = np.array([int(h), int(w)])
    target["size"] = np.array([int(h), int(w)])

    return image, target


def make_coco_transforms(image_set, cfg):
    """make coco transforms"""
    scales = cfg.img_scales
    max_size = cfg.max_img_size
    scales_val = [scales[-1]]
    if image_set == 'train':
        return src_transforms.Compose([
            src_transforms.RandomHorizontalFlip(),
            src_transforms.RandomSelect(
                src_transforms.RandomResize(scales, max_size=max_size),
                src_transforms.Compose([
                    src_transforms.RandomResize([400, 500, 600]),
                    src_transforms.RandomSizeCrop(384, 600),
                    src_transforms.RandomResize(scales, max_size=max_size),
                ])
            ),
            src_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if image_set == 'val':
        return src_transforms.Compose([
            src_transforms.RandomResize(scales_val, max_size=max_size),
            src_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    raise ValueError(f'unknown {image_set}')


def pad_image_to_max_size(img, max_size=1333):
    """pad image to max size"""
    c, h, w = img.shape
    mask = np.ones((max_size, max_size), dtype=np.bool)
    pad_img = np.zeros((c, max_size, max_size))
    pad_img[: c, : h, : w] = img
    mask[: h, : w] = False
    return pad_img.astype('float32'), mask.astype('float32')


def build_dataset(cfg):
    """build dataset"""
    if cfg.eval:
        image_set = 'val'
    else:
        image_set = 'train'
    root = Path(cfg.coco_path)
    mode = 'instances'
    paths = {
        "train": (root / "images/train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "images/val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = paths[image_set]
    dataset = CocoDataset(img_folder,
                          ann_file,
                          num_queries=cfg.num_queries,
                          transforms=make_coco_transforms(image_set, cfg),
                          num_classes=cfg.num_classes)
    base_ds = dataset.coco
    length_dataset = len(dataset)
    column_names = ['image', 'bboxes', 'labels', 'orig_sizes', 'n_boxes', 'img_id']
    sampler = ds.DistributedSampler(cfg.device_num, cfg.rank)
    dataset = ds.GeneratorDataset(dataset,
                                  column_names=column_names,
                                  num_parallel_workers=cfg.num_workers,
                                  max_rowsize=20,
                                  sampler=sampler)
    dataset = dataset.map(
        partial(pad_image_to_max_size, max_size=cfg.max_img_size),
        input_columns=['image'],
        output_columns=['image', 'mask'],
        column_order=['image', 'mask', 'bboxes', 'labels', 'orig_sizes', 'n_boxes', 'img_id'],
        num_parallel_workers=cfg.num_workers
    )
    if cfg.eval:
        dataset = dataset.batch(cfg.batch_size)
        dataset = dataset.repeat(1)
        return dataset, base_ds, length_dataset
    dataset = dataset.batch(cfg.batch_size, drop_remainder=True)
    dataset = dataset.repeat(cfg.epochs)
    return dataset, length_dataset
