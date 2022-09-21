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
"""COCO_dataset"""
import os
import random
import cv2
import numpy as np
import mindspore.dataset as de
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.c_transforms as c_vision

from pycocotools.coco import COCO
from PIL import Image

def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class COCODataset:
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, dataset_dir, annotation_file, resize_size, is_train=True, transform=None):
        if resize_size is None:
            resize_size = [800, 1333]
        self.coco = COCO(annotation_file)
        self.root = dataset_dir
        ids = list(sorted(self.coco.imgs.keys()))
        new_ids = []
        for i in ids:
            ann_id = self.coco.getAnnIds(imgIds=i, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                new_ids.append(i)
        self.ids = new_ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.transform = transform
        self.resize_size = resize_size
        self.train = is_train

    def getImg(self, index):
        img_id = self.ids[index]
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        return img, target

    def __getitem__(self, index):
        img, ann = self.getImg(index)
        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.transform is not None:
                img, boxes = self.transform(img, boxes)
        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]
        to_tensor = py_vision.ToTensor()
        img = to_tensor(img)
        max_h = 1344
        max_w = 1344
        max_num = 90
        img = np.pad(img, ((0, 0), (0, max(int(max_h - img.shape[1]), 0)), (0, max(int(max_w - img.shape[2]), 0))))
        normalize_op = c_vision.Normalize(mean=[0.40789654, 0.44719302, 0.47026115], \
        std=[0.28863828, 0.27408164, 0.27809835])
        img = img.transpose(1, 2, 0)    # chw to hwc
        img = normalize_op(img)
        img = img.transpose(2, 0, 1)     #hwc to chw
        boxes = np.pad(boxes, ((0, max(max_num-boxes.shape[0], 0)), (0, 0)), 'constant', constant_values=-1)
        classes = np.pad(classes, (0, max(max_num - len(classes), 0)), 'constant', constant_values=-1).astype('int32')

        return img, boxes, classes

    def __len__(self):
        return len(self.ids)

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32
        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized
        if boxes is not None:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
        return image_paded, boxes

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if annot is None:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True
def create_coco_dataset(dataset_dir, annotation_file, batch_size, shuffle=True, \
                        transform=None, num_parallel_workers=8, num_shards=None, shard_id=None):
    cv2.setNumThreads(0)
    dataset = COCODataset(dataset_dir, annotation_file, is_train=True, transform=transform)
    dataset_column_names = ["img", "boxes", "class"]
    ds = de.GeneratorDataset(dataset, column_names=dataset_column_names, \
    shuffle=shuffle, num_parallel_workers=min(8, num_parallel_workers), num_shards=num_shards, shard_id=shard_id)
    ds = ds.batch(batch_size, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    return ds, len(dataset)
