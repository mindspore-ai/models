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
"""FasterRcnn Ssod dataset"""

import os
import copy
import random
import operator
import logging

import numpy as np
from PIL import Image, ImageFilter
import mindspore.dataset as ms_dataset
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.py_transforms as P2
from mindspore.communication.management import get_rank, get_group_size
from pycocotools.coco import COCO


class GaussianBlur:
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_strong_augmentation(is_train):
    """
    Inputs:
        PIL Image of shape (H, W, C) in the range [0, 255]
    Outputs:
        numpy.ndarray of shape (C, H, W) in the range [0.0, 1.0]
    """

    augmentation = []
    if is_train:
        augmentation.append(P2.RandomApply([P.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)], prob=0.8))
        augmentation.append(P.RandomGrayscale(prob=0.2))
        augmentation.append(P2.RandomApply([GaussianBlur([0.1, 2.0])], prob=0.5))

        randcrop_transform = P2.Compose(
            [
                P.ToTensor(),
                P.Normalize((123.675 / 255, 116.28 / 255, 103.53 / 255), (58.395 / 255, 57.12 / 255, 57.375 / 255)),
                P.RandomErasing(prob=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
                P.RandomErasing(prob=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
                P.RandomErasing(prob=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
            ]
        )
        augmentation.append(randcrop_transform)

    return P2.Compose(augmentation)


def build_weak_augmentation():
    """
    Inputs:
        PIL Image of shape (H, W, C) in the range [0, 255]
    Outputs:
        numpy.ndarray of shape (C, H, W) in the range [0.0, 1.0]
    """

    augmentation = [
        P.ToTensor(),
        P.Normalize((123.675 / 255, 116.28 / 255, 103.53 / 255), (58.395 / 255, 57.12 / 255, 57.375 / 255)),
    ]
    return P2.Compose(augmentation)


class PreprocessFn:
    """preprocess function for mindspore dataset of fasterrcnn net"""

    def __init__(self, cfg, is_training=True):
        self.cfg = cfg
        self.is_training = is_training
        self.resize_fn = P.Resize(size=(self.cfg.img_height, self.cfg.img_width))
        self.weak_augmentation = build_weak_augmentation()
        if is_training:
            self.strong_augmentation = build_strong_augmentation(is_train=is_training)

    def __call__(self, *inputs):
        if self.is_training:
            label_img, label_annos, unlabel_img, unlabel_annos = inputs
            label_img_strong, label_img_weak, label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums = \
                self.process(label_img, label_annos)
            unlabel_img_strong, unlabel_img_weak, unlabel_img_metas, unlabel_gt_bboxes, unlabel_gt_labels, \
                unlabel_gt_nums = self.process(unlabel_img, unlabel_annos)
            return label_img_strong, label_img_weak, \
                   label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums, \
                   unlabel_img_strong, unlabel_img_weak, \
                   unlabel_img_metas, unlabel_gt_bboxes, unlabel_gt_labels, unlabel_gt_nums
        label_img, label_annos = inputs
        label_img_weak, label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums = \
            self.process(label_img, label_annos)
        return label_img_weak, label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums

    def process(self, img, annos):
        img, img_metas, gt_bboxes, gt_labels, gt_nums = PreprocessFn.format_inputs(img, annos)
        if self.is_training:
            img_weak_pil, img_weak, img_metas, gt_bboxes = self.weak_process(img, gt_bboxes)
            img_strong = self.strong_process(img_weak_pil)
            return img_strong, img_weak, img_metas, gt_bboxes, gt_labels, gt_nums
        _, img_weak, img_metas, gt_bboxes = self.weak_process(img, gt_bboxes)
        return img_weak, img_metas, gt_bboxes, gt_labels, gt_nums

    @staticmethod
    def format_inputs(img, annos):
        img_metas = img.shape[:2]
        gt_bboxes = annos[:, :4]
        gt_labels = annos[:, 4]
        gt_nums = annos[:, 5]

        pad_max_number = 128
        gt_bboxes_new = np.pad(gt_bboxes, ((0, pad_max_number - annos.shape[0]), (0, 0)),
                               mode="constant", constant_values=0)
        gt_labels_new = np.pad(gt_labels, ((0, pad_max_number - annos.shape[0])), mode="constant", constant_values=-1)
        gt_labels_new = gt_labels_new.astype(np.int32)
        gt_nums_new = np.pad(gt_nums, ((0, pad_max_number - annos.shape[0])), mode="constant", constant_values=1)
        gt_nums_new_revert = ~(gt_nums_new.astype(np.bool))
        return img, img_metas, gt_bboxes_new, gt_labels_new, gt_nums_new_revert

    @staticmethod
    def rescale_with_tuple(img, scale):
        height, width = img.shape[:2]
        scale_factor = min(max(scale) / max(height, width), min(scale) / min(height, width))
        new_size = int(width * float(scale_factor) + 0.5), int(height * float(scale_factor) + 0.5)

        pil_image = Image.fromarray(img.astype("uint8"), "RGB")
        pil_image = pil_image.resize(new_size, Image.BILINEAR)
        rescaled_img = np.array(pil_image)
        return rescaled_img, scale_factor

    def weak_process(self, img, gt_bboxes):
        """resize and flip"""
        ori_h, ori_w = img.shape[:2]

        if self.cfg.keep_ratio:
            # rescale
            img_data, scale_factor = PreprocessFn.rescale_with_tuple(img, (self.cfg.img_width, self.cfg.img_height))
            if img_data.shape[0] > self.cfg.img_height:
                img_data, scale_factor_2 = PreprocessFn.rescale_with_tuple(img_data,
                                                                           (self.cfg.img_height, self.cfg.img_height))
                scale_factor = scale_factor * scale_factor_2

            pad_h = self.cfg.img_height - img_data.shape[0]
            pad_w = self.cfg.img_width - img_data.shape[1]
            if not ((pad_h >= 0) and (pad_w >= 0)):
                raise ValueError("[ERROR] rescale h, w = {}, {} failed.".format(img_data.shape[0], img_data.shape[1]))
            pad_img_data = np.zeros((self.cfg.img_height, self.cfg.img_width, 3)).astype(img_data.dtype)
            pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

            img_pil = Image.fromarray(pad_img_data.astype("uint8"), "RGB")
            scale_h = scale_factor
            scale_w = scale_factor
        else:
            # resize
            img_pil = Image.fromarray(img.astype("uint8"), "RGB")
            img_pil = self.resize_fn(img_pil)
            scale_h = self.cfg.img_height / ori_h
            scale_w = self.cfg.img_width / ori_w

        if self.is_training:
            img_metas = (self.cfg.img_height, self.cfg.img_width, 1.0)
        else:
            img_metas = (ori_h, ori_w, scale_h, scale_w)
        img_metas = np.asarray(img_metas, dtype=np.float32)

        gt_bboxes[:, 0::2] = gt_bboxes[:, 0::2] * scale_w
        gt_bboxes[:, 1::2] = gt_bboxes[:, 1::2] * scale_h
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_metas[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_metas[0] - 1)

        # flip
        flip = (np.random.rand() < self.cfg.flip_ratio)
        if self.is_training and flip:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            gt_bboxes_old = gt_bboxes.copy()
            gt_bboxes[..., 0::4] = img_metas[1] - gt_bboxes_old[..., 2::4] - 1
            gt_bboxes[..., 2::4] = img_metas[1] - gt_bboxes_old[..., 0::4] - 1

        img_pil_new = img_pil.copy()
        img_weak = self.weak_augmentation(img_pil_new)[0]
        gt_bboxes = gt_bboxes.astype(np.float32)
        return img_pil, img_weak, img_metas, gt_bboxes

    def strong_process(self, img_pil):
        img_strong = self.strong_augmentation(img_pil)[0]
        return img_strong


class SemisupCocoDataset:
    """label and unlabel, two part dataset for semisup training"""

    def __init__(self, is_training: bool, img_dir: str, ann_file: str):
        self.is_training = is_training
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        # modelzoo needs increase the category range by one
        self.cat_ids_to_continuous_ids = {cat_id: idx + 1 for idx, cat_id in enumerate(self.coco.getCatIds())}

        self.img_ids = self.coco.getImgIds()
        logging.info("Loaded %d images in COCO format from %s", len(self.img_ids), ann_file)
        if is_training:
            self.img_ids = self.filter_images_with_only_crowd_annotations(self.img_ids)

        self.label_img_ids, self.unlabel_img_ids = self.divide_label_unlabel(dataset_ids=self.img_ids)
        self.label_img_ids_size = len(self.label_img_ids)
        self.unlabel_img_ids_size = len(self.unlabel_img_ids)
        logging.info("label_dataset size: %d, unlabel_dataset size: %d",
                     self.label_img_ids_size, self.unlabel_img_ids_size)
        if is_training and (self.label_img_ids_size == 0 or self.unlabel_img_ids_size == 0):
            raise ValueError("[ERROR] label_dataset size: {} or unlabel_dataset size: {} is zero"
                             .format(self.label_img_ids_size, self.unlabel_img_ids_size))

    def __len__(self):
        return max(self.label_img_ids_size, self.unlabel_img_ids_size)

    def __getitem__(self, idx):
        label_img_id = self.label_img_ids[idx % self.label_img_ids_size]
        label_img, label_annos = self.get_img_and_annos(label_img_id)

        if self.is_training:
            unlabel_img_id = self.unlabel_img_ids[idx % self.unlabel_img_ids_size]
            unlabel_img, unlabel_annos = self.get_img_and_annos(unlabel_img_id)
            return label_img, label_annos, unlabel_img, unlabel_annos
        return label_img, label_annos, label_img_id

    def get_img_and_annos(self, img_id: int):
        """
        :param img_id: int, image id in coco type dataset
        :return: img: np.array from PIL.Image RGB
                 img_annos: np array
        """
        img_path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB")
        img = np.array(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = copy.deepcopy(self.coco.loadAnns(ann_ids))
        img_annos = self.create_coco_label(annotations)
        return img, img_annos

    def create_coco_label(self, annotations):
        annos = []
        for anno in annotations:
            bbox = anno["bbox"]
            annos.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] +
                         [self.cat_ids_to_continuous_ids[anno["category_id"]] if anno["category_id"] != -1 else -1] +
                         [int(anno["iscrowd"])])
        if annos:
            img_annos = np.array(annos, dtype=np.int32)
        else:
            img_annos = np.array([[0, 0, 0, 0, 0, 1]], dtype=np.int32)
        return img_annos

    def divide_label_unlabel(self, dataset_ids):
        if not self.is_training:
            return dataset_ids, list()

        # process new type annotation file which includes label and unlabel info
        ann_ids = self.coco.getAnnIds(imgIds=dataset_ids)
        annotations = self.coco.loadAnns(ann_ids)

        unlabels_cmp_category_id = -1
        unlabels_cmp_bbox = [0 for _ in range(4)]
        labels = []
        unlabels = []
        for anno in annotations:
            if anno["category_id"] == unlabels_cmp_category_id and operator.eq(anno["bbox"], unlabels_cmp_bbox):
                unlabels.append(anno["image_id"])
            else:
                labels.append(anno["image_id"])

        labels = list(set(labels))
        unlabels = list(set(unlabels))
        return labels, unlabels

    def filter_images_with_only_crowd_annotations(self, img_ids):
        """
        Filter out images with none annotations or only crowd annotations
        (i.e., images without non-crowd annotations).
        A common training-time preprocessing on COCO dataset.

        Args:
            img_ids (list[int]): img ids list.

        Returns:
            list[int]: the same format, but filtered.
        """
        num_before = len(img_ids)

        def valid(img_id):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann.get("iscrowd", 0) == 0:
                    return True
            return False

        img_ids = [img_id for img_id in img_ids if valid(img_id)]
        num_after = len(img_ids)
        logging.info("Removed %d images with no usable annotations. %d images left.",
                     num_before - num_after, num_after)
        return img_ids


def create_semisup_dataset(cfg, is_training=True):
    if is_training:
        img_dir = cfg.train_img_dir
        ann_file = cfg.train_ann_file
    else:
        img_dir = cfg.eval_img_dir
        ann_file = cfg.eval_ann_file

    semisup_dataset = SemisupCocoDataset(is_training=is_training, img_dir=img_dir, ann_file=ann_file)
    preprocess_fn = PreprocessFn(cfg, is_training=is_training)

    # get rank_id and rank_size
    rank_id, rank_size = None, None
    if cfg.run_distribute:
        rank_id = get_rank()
        rank_size = get_group_size()

    if is_training:
        data_loader = ms_dataset.GeneratorDataset(semisup_dataset,
                                                  column_names=["label_img", "label_annos",
                                                                "unlabel_img", "unlabel_annos"],
                                                  num_parallel_workers=cfg.num_parallel_workers,
                                                  shuffle=is_training,
                                                  num_shards=rank_size, shard_id=rank_id,
                                                  max_rowsize=12)
        data_loader = data_loader.map(operations=preprocess_fn,
                                      input_columns=["label_img", "label_annos", "unlabel_img", "unlabel_annos"],
                                      output_columns=["label_img_strong", "label_img_weak", "label_img_metas",
                                                      "label_gt_bboxes", "label_gt_labels", "label_gt_nums",
                                                      "unlabel_img_strong", "unlabel_img_weak", "unlabel_img_metas",
                                                      "unlabel_gt_bboxes", "unlabel_gt_labels", "unlabel_gt_nums"],
                                      column_order=["label_img_strong", "label_img_weak", "label_img_metas",
                                                    "label_gt_bboxes", "label_gt_labels", "label_gt_nums",
                                                    "unlabel_img_strong", "unlabel_img_weak", "unlabel_img_metas",
                                                    "unlabel_gt_bboxes", "unlabel_gt_labels", "unlabel_gt_nums"],
                                      num_parallel_workers=cfg.num_parallel_workers)
        data_loader = data_loader.batch(cfg.batch_size, drop_remainder=True)
        data_loader = data_loader.repeat(-1)
    else:
        data_loader = ms_dataset.GeneratorDataset(semisup_dataset,
                                                  column_names=["label_img", "label_annos", "label_img_id"],
                                                  num_parallel_workers=cfg.num_parallel_workers,
                                                  shuffle=is_training,
                                                  num_shards=rank_size, shard_id=rank_id,
                                                  max_rowsize=12)
        data_loader = data_loader.map(operations=preprocess_fn,
                                      input_columns=["label_img", "label_annos"],
                                      output_columns=["label_img_weak", "label_img_metas", "label_gt_bboxes",
                                                      "label_gt_labels", "label_gt_nums"],
                                      column_order=["label_img_weak", "label_img_metas", "label_gt_bboxes",
                                                    "label_gt_labels", "label_gt_nums", "label_img_id"],
                                      num_parallel_workers=cfg.num_parallel_workers)
        data_loader = data_loader.batch(cfg.test_batch_size, drop_remainder=False)

    return data_loader
