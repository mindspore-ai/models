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
import random
import cv2
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P

from src.model_utils.config import config
from src.maskrcnn.mask_rcnn_r50 import Mask_Rcnn_Resnet50
from src.model_utils.device_adapter import get_device_id
random.seed(1)
ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

def rescale_with_tuple(img, scale):
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor

def rescale_column_test(img, img_shape):
    """rescale operation for image of eval"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor * scale_factor2

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = np.append(img_shape, (scale_factor, scale_factor))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return pad_img_data, img_shape

def imnormalize_column(img):
    """imnormalize operation for image"""
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    return img_data

def transpose_column(img, img_shape):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    return img_data, img_shape

def process_img(img):
    """process image"""
    # rescale image
    img_shape = img.shape[:2]
    img, img_shape = rescale_column_test(img, img_shape)
    img = imnormalize_column(img)
    img, img_shape = transpose_column(img, img_shape)
    img = Tensor(img)
    img_shape = Tensor(img_shape)
    expand_dim = P.ExpandDims()
    img = expand_dim(img, 0)
    img_shape = expand_dim(img_shape, 0)
    return img, img_shape

def save_result(img, boxes, labels, img_metas_, save_name):
    """save the detection result"""
    num_classes = config.num_classes
    classes_name = config.coco_classes
    color_list = []
    for _ in range(num_classes):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_list.append((r, g, b))
    for k in range(len(labels)):
        box = boxes[k].tolist()
        label = labels[k].tolist()
        if box[-1] > 0.5 and label < num_classes:
            [x1, y1, x2, y2] = [int(box[l]) for l in range(len(box) - 1)]
            w, h = x2 - x1, y2 - y1
            image_height, image_width = int(img_metas_[0][0]), int(img_metas_[0][1])
            if x2 > image_width or y2 > image_height or w <= 0 or h <= 0:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), color_list[label], thickness=2)
            text = classes_name[label + 1]
            cv2.putText(img, text, (x1, int(y1*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if config.vis_result:
            cv2.imshow('res', img)
            cv2.waitKey(0)
    cv2.imwrite(save_name, img)

def det():
    net = Mask_Rcnn_Resnet50(config)
    param_dict = ms.load_checkpoint(config.ckpt_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)
    image_list = os.listdir(config.image_folder)
    max_num = config.num_gts
    for image_name in tqdm(image_list):
        img = cv2.imread(os.path.join(config.image_folder, image_name))
        img_data, img_metas = process_img(img)
        output = net(img_data, img_metas, None, None, None, None)
        bbox = output[0]
        label = output[1]
        mask = output[2]

        bbox = np.squeeze(bbox.asnumpy()[0, :, :])
        label = np.squeeze(label.asnumpy()[0, :, :])
        mask = np.squeeze(mask.asnumpy()[0, :, :])

        bbox = bbox[mask, :]
        label = label[mask]

        if bbox.shape[0] > max_num:
            inds = np.argsort(-bbox[:, -1])
            inds = inds[:max_num]
            bbox = bbox[inds]
            label = label[inds]
        save_result(img, bbox, label, img_metas,
                    os.path.join(config.save_result_folder, image_name))

if __name__ == '__main__':
    det()
