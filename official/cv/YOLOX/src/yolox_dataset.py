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
# =======================================================================================
""" Yolox dataset module """
import multiprocessing
import random
import os
import math

import numpy as np
import cv2
import mindspore.dataset as de
from pycocotools.coco import COCO

from src.transform import random_affine, TrainTransform, ValTransform

min_keypoints_per_image = 10


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def has_valid_annotation(anno):
    """Check annotation file."""
    # if it's empty, there is no annotation
    if not anno:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    """ Get mosaic coordinate """
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


class COCOYoloXDataset:
    """ YoloX Dataset for COCO """

    def __init__(self, root, ann_file, remove_images_without_annotations=True,
                 filter_crowd_anno=True, is_training=True, mosaic=True, img_size=(640, 640),
                 preproc=None, input_dim=(640, 640), mosaic_prob=1.0, enable_mosaic=True, enable_mixup=True,
                 mixup_prob=1.0, eval_parallel=False, device_num=1, batch_size=8):
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.filter_crowd_anno = filter_crowd_anno
        self.is_training = is_training
        self.root = root
        self.mosaic = mosaic
        self.img_size = img_size
        self.preproc = preproc
        self.input_dim = input_dim
        self.mosaic_prob = mosaic_prob
        self.enable_mosaic = enable_mosaic
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2.0)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.mixup_prob = mixup_prob
        self.enable_mixup = enable_mixup

        if remove_images_without_annotations:
            img_ids = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    img_ids.append(img_id)
            self.img_ids = img_ids
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}
        self.cat_ids_to_continuous_ids = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.continuous_ids_cat_ids = {v: k for k, v in self.cat_ids_to_continuous_ids.items()}

        if not is_training and eval_parallel:
            from model_utils.config import config
            step_img_size = device_num * batch_size
            append_img_num = math.ceil(len(self.img_ids) / step_img_size) * step_img_size - len(self.img_ids)
            if append_img_num > 0:
                config.logger.info(f"[INFO]: Dataset size {len(self.img_ids)} is not divisible "
                                   f"by step_img_size {step_img_size}. Append duplicate images.")
                self.img_ids.extend(self.img_ids[:append_img_num])

    def pull_item(self, index):
        """
        pull image and label
        """
        res, img_info, _ = self.load_anno_from_ids(index)
        img = self.load_resized_img(index)
        return img, res.copy(), img_info, np.array([self.img_ids[index]])

    def mosaic_proc(self, idx):
        """ Mosaic data augment """
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self.input_dim
            input_h, input_w = input_dim[0], input_dim[1]
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
            for i_mosaic, index in enumerate(indices):
                img, _labels, _, _ = self.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if mosaic_labels:
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            if (
                    self.enable_mixup
                    and not mosaic_labels.size == 0
                    and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            mix_img, padded_labels, pre_fg_mask, is_inbox_and_incenter = self.preproc(mosaic_img, mosaic_labels,
                                                                                      self.input_dim)
            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, pre_fg_mask, is_inbox_and_incenter
        img, label, _, _ = self.pull_item(idx)
        img, label, pre_fg_mask, is_inbox_and_incenter = self.preproc(img, label, self.input_dim)
        return img, label, pre_fg_mask, is_inbox_and_incenter

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while not cp_labels:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self.load_anno_from_ids(cp_index)
        img, cp_labels, _, _ = self.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def load_anno_from_ids(self, index):
        """
        load annotations via ids
        """
        img_id = self.img_ids[index]
        im_ann = self.coco.loadImgs(img_id)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
        nums_objs = len(objs)
        res = np.zeros((nums_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.cat_ids_to_continuous_ids[obj["category_id"]]
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        img_info = (height, width)
        resize_info = (int(height * r), int(width * r))
        return res, img_info, resize_info

    def load_resized_img(self, index):
        """
        resize to fix size
        """
        img_id = self.img_ids[index]
        img_path = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.root, img_path)
        img = cv2.imread(img_path)
        img = np.array(img)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resize_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resize_img

    def __getitem__(self, index):
        if self.is_training:
            img, labels, pre_fg_mask, is_inbox_and_incenter = self.mosaic_proc(index)
            return img, labels, pre_fg_mask, is_inbox_and_incenter
        img, _, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, _ = self.preproc(img, self.input_dim)
            img = img.astype(np.float32)
        return img, img_info, img_id

    def __len__(self):
        return len(self.img_ids)


def create_yolox_dataset(image_dir, anno_path, batch_size, device_num, rank,
                         data_aug=True, is_training=True):
    """ create yolox dataset """
    from model_utils.config import config
    cv2.setNumThreads(0)
    if is_training:
        filter_crowd = False
        remove_empty_anno = False
    else:
        filter_crowd = False
        remove_empty_anno = False
    img_size = config.input_size
    input_dim = img_size
    if is_training:
        yolo_dataset = COCOYoloXDataset(root=image_dir, ann_file=anno_path, filter_crowd_anno=filter_crowd,
                                        remove_images_without_annotations=remove_empty_anno, is_training=is_training,
                                        mosaic=data_aug, enable_mixup=data_aug, enable_mosaic=data_aug,
                                        preproc=TrainTransform(config=config), img_size=img_size, input_dim=input_dim)
    else:
        yolo_dataset = COCOYoloXDataset(
            root=image_dir, ann_file=anno_path, filter_crowd_anno=filter_crowd,
            remove_images_without_annotations=remove_empty_anno, is_training=is_training, mosaic=False,
            enable_mixup=False,
            img_size=img_size, input_dim=input_dim, preproc=ValTransform(legacy=False),
            device_num=device_num, batch_size=batch_size, eval_parallel=config.eval_parallel)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)
    if is_training:
        dataset_column_names = ["image", "labels", "pre_fg_mask", "is_inbox_and_inCenter"]
        ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names,
                                 num_parallel_workers=min(8, num_parallel_workers),
                                 python_multiprocessing=True,
                                 shard_id=rank, num_shards=device_num, shuffle=True)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:  # for val
        if config.eval_parallel:
            ds = de.GeneratorDataset(yolo_dataset, column_names=["image", "image_shape", "img_id"],
                                     num_parallel_workers=min(8, num_parallel_workers), shuffle=False,
                                     python_multiprocessing=True, shard_id=rank, num_shards=device_num)
        else:
            ds = de.GeneratorDataset(yolo_dataset, column_names=["image", "image_shape", "img_id"],
                                     num_parallel_workers=min(8, num_parallel_workers), shuffle=False)
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.repeat(1)
    return ds
