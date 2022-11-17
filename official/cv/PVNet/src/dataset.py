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
"""training dataset"""
import os

import cv2
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import numpy as np

from model_utils.config import config as cfg


def crop_or_padding_to_fixed_size(img, mask, target_height, target_width):
    """crop_or_padding_to_fixed_size"""
    h, w, _ = img.shape
    hpad, wpad = target_height >= h, target_width >= w

    hbeg = 0 if hpad else np.random.randint(0, h - target_height)
    wbeg = 0 if wpad else np.random.randint(0,
                                            w - target_width)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    hend = hbeg + target_height
    wend = wbeg + target_width

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([target_height, target_width, 3], dtype=img.dtype)
        new_mask = np.zeros([target_height, target_width], dtype=mask.dtype)

        hbeg = 0 if not hpad else (target_height - h) // 2
        wbeg = 0 if not wpad else (target_width - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask

        img, mask = new_img, new_mask

    return img, mask


def crop_or_padding_to_fixed_size_instance(img, mask, hcoords, th, tw, overlap_ratio=0.5):
    """crop_or_padding_to_fixed_size_instance"""
    h, w, _ = img.shape
    hs, ws = np.nonzero(mask)

    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax = np.min(ws), np.max(ws)
    fh, fw = hmax - hmin, wmax - wmin
    hpad, wpad = th >= h, tw >= w

    hrmax = int(min(hmin + overlap_ratio * fh, h - th))  # h must > target_height else hrmax<0
    hrmin = int(max(hmin + overlap_ratio * fh - th, 0))
    wrmax = int(min(wmin + overlap_ratio * fw, w - tw))  # w must > target_width else wrmax<0
    wrmin = int(max(wmin + overlap_ratio * fw - tw, 0))

    hbeg = 0 if hpad else np.random.randint(hrmin, hrmax)
    hend = hbeg + th
    wbeg = 0 if wpad else np.random.randint(wrmin,
                                            wrmax)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]

    hcoords[:, 0] -= wbeg * hcoords[:, 2]
    hcoords[:, 1] -= hbeg * hcoords[:, 2]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask
        hcoords[:, 0] += wbeg * hcoords[:, 2]
        hcoords[:, 1] += hbeg * hcoords[:, 2]

        img, mask = new_img, new_mask

    return img, mask, hcoords


def rotate_instance(img, mask, hcoords, rot_ang_min, rot_ang_max):
    """rotate_instance"""
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    hs, ws = np.nonzero(mask)
    R = cv2.getRotationMatrix2D((np.mean(ws), np.mean(hs)), degree, 1)
    mask = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    last_row = np.asarray([[0, 0, 1]], np.float32)
    hcoords = np.matmul(hcoords, np.concatenate([R, last_row], 0).transpose())
    return img, mask, hcoords


def compute_vertex_hcoords(mask, hcoords, use_motion=False):
    """compute_vertex_hcoords"""
    h, w = mask.shape
    m = hcoords.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :] * hcoords[None, :, 2:]
    vertex = hcoords[None, :, :2] - vertex
    if not use_motion:
        norm = np.linalg.norm(vertex, axis=2, keepdims=True)
        norm[norm < 1e-3] += 1e-3
        vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return np.reshape(vertex_out, [h, w, m * 2])


def crop_resize_instance_v1(img, mask, hcoords, crop_height, crop_width,
                            overlap_ratio=0.5, ratio_min=0.8, ratio_max=1.2):
    """
    crop a region with [imheight*resize_ratio,imwidth*resize_ratio]
    which at least overlap with foreground bbox with overlap
    """
    scale_ratio = np.random.uniform(1, 2)
    scale_height = int(img.shape[0] * scale_ratio)
    scale_width = int(img.shape[1] * scale_ratio)
    img = cv2.resize(img, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (scale_width, scale_height), interpolation=cv2.INTER_NEAREST)

    hcoords[:, 0] = hcoords[:, 0] * scale_ratio
    hcoords[:, 1] = hcoords[:, 1] * scale_ratio

    resize_ratio = np.random.uniform(ratio_min, ratio_max)
    target_height = int(crop_height * resize_ratio)
    target_width = int(crop_width * resize_ratio)

    img, mask, hcoords = crop_or_padding_to_fixed_size_instance(
        img, mask, hcoords, target_height, target_width, overlap_ratio)

    img = cv2.resize(img, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)

    hcoords[:, 0] = hcoords[:, 0] / resize_ratio
    hcoords[:, 1] = hcoords[:, 1] / resize_ratio

    return img, mask, hcoords


def augmentation(img, mask, hcoords):
    """augmentation"""
    foreground = np.sum(mask)
    if foreground > 0:
        if cfg.rotation:
            img, mask, hcoords = rotate_instance(
                img, mask, hcoords,
                cfg.rot_ang_min, cfg.rot_ang_max
            )

        if cfg.crop:
            img, mask, hcoords = crop_resize_instance_v1(
                img, mask, hcoords, cfg.img_crop_size_height, cfg.img_crop_size_width, cfg.overlap_ratio,
                cfg.resize_ratio_min, cfg.resize_ratio_max
            )
    else:
        img, mask = crop_or_padding_to_fixed_size(img, mask, cfg.img_crop_size_height, cfg.img_crop_size_width)
    return img, mask, hcoords


def blur_image(img, sigma=3):
    """blur_image"""
    return cv2.GaussianBlur(img, (sigma, sigma), 0)


def preprocess_fn(_image, _mask, _farthest):
    """preprocess_fn"""
    rgb = np.fromstring(_image, dtype=np.uint8).reshape(cfg.img_height, cfg.img_width, 3) # 480x640x3
    mask = np.frombuffer(_mask, dtype=np.int32).reshape(cfg.img_height, cfg.img_width) # 480x640

    if mask.max() == 255:
        mask = np.asarray(mask, np.float32) / 255.0
    mask = np.round(mask).astype(np.uint8)

    hcoords = np.reshape(_farthest, [cfg.vote_num, 2])
    hcoords = np.concatenate([hcoords, np.ones([hcoords.shape[0], 1], np.float32)], 1)

    rgb, mask, hcoords = augmentation(rgb, mask, hcoords)
    vertex = compute_vertex_hcoords(mask, hcoords, False)
    vertex_weight = np.ascontiguousarray(mask, dtype=np.float32)
    vertex_weight = np.expand_dims(vertex_weight, 0)
    mask = mask.astype(np.int32)
    if np.random.random() < 0.5:
        blur_image(rgb, np.random.choice([3, 5, 7, 9]))

    return rgb, mask, vertex.transpose([2, 0, 1]), vertex_weight


def create_dataset(cls_list, batch_size=16, workers=16, devices=1, rank=0, multi_process=True, training=True):
    """
    Create a train or eval dataset.
    Args:
        cls_list: (str): object class name
        batch_size (int): The batch size of dataset. Default: 16.
        workers: (int): the number of wokers for data preprocessing
        devices: (int): the number of gpu card, default : 1
        rank: (int): the current index of gpu cards
        multi_process: (boo): enable multi threads for data preprocessing
        training: (bool): training stage or test stage
    Returns:
        Dataset.
    """

    mind_ds_path = os.path.join(cfg.data_url, cls_list, "pvnettrain.mindrecord0")
    print(mind_ds_path)
    ds = de.MindDataset(mind_ds_path,
                        columns_list=["image", "mask", "farthest"],
                        num_shards=devices,
                        shard_id=rank,
                        shuffle=training)

    ds = ds.map(input_columns=["image", "mask", "farthest"],
                output_columns=["image", "mask", "vertex", "vertex_weight"],
                operations=preprocess_fn, num_parallel_workers=workers, python_multiprocessing=multi_process)

    img_transforms = C.Compose([
        CV.RandomColorAdjust(
            cfg.brightness, cfg.contrast,
            cfg.saturation, cfg.hue),
        CV.ToTensor(),  # 0~255 HWC to 0~1 CHW
        C.TypeCast(mstype.float32),
        # Computed from random subset of ImageNet training images
        CV.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False),
    ])

    mask_transforms = [
        C.TypeCast(mstype.int32),
    ]
    ver_transforms = [
        C.TypeCast(mstype.float32),
    ]
    vertex_weight_transforms = [
        C.TypeCast(mstype.float32),
    ]
    ds = ds.map(operations=img_transforms,
                input_columns='image',
                num_parallel_workers=workers,
                python_multiprocessing=multi_process)
    ds = ds.map(operations=mask_transforms,
                input_columns='mask',
                num_parallel_workers=workers,
                python_multiprocessing=multi_process)
    ds = ds.map(operations=ver_transforms,
                input_columns='vertex',
                num_parallel_workers=workers,
                python_multiprocessing=multi_process)
    ds = ds.map(operations=vertex_weight_transforms,
                input_columns='vertex_weight',
                num_parallel_workers=workers,
                python_multiprocessing=multi_process)

    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_workers=workers)
    return ds
