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
"""
YOLOv3 based on DarkNet.
"""
import sys
import numpy as np
from PIL import Image


def load_image(img_path, image_size):
    img = Image.open(img_path).convert("RGB")
    img, ori_img_shape = reshape_fn(img, image_size)
    img = np.transpose(img, (2, 0, 1))
    img = np.array([img])
    return img, ori_img_shape

def pil_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]

def get_interp_method(interp, sizes=()):
    '''
    get_interp_method
    '''
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp

def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img/255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return img

def _reshape_data(image, image_size):
    """Reshape image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    ori_w, ori_h = image.size
    ori_image_shape = np.array([ori_w, ori_h], np.int32)
    # original image shape fir:H sec:W
    h, w = image_size
    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, h, w))
    image = image.resize((w, h), pil_image_reshape(interp))
    image_data = statistic_normalize_img(image, statistic_norm=True)
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
    image_data = image_data.astype(np.float32)
    return image_data, ori_image_shape

def reshape_fn(image, input_size):
    image, ori_image_shape = _reshape_data(image, image_size=input_size)
    return image, ori_image_shape

def detect(outputs, image_shape, threshold):
    '''
    detect
    '''
    bboxes = []
    for out_item_single in outputs:
           # get number of items in one head, [B, gx, gy, anchors, 5+80]
        dimensions = out_item_single.shape[:-1]
        out_num = 1
        for d in dimensions:
            out_num *= d
        ori_w, ori_h = image_shape

        x = out_item_single[..., 0] * ori_w
        y = out_item_single[..., 1] * ori_h
        w = out_item_single[..., 2] * ori_w
        h = out_item_single[..., 3] * ori_h

        conf = out_item_single[..., 4:5]
        cls_emb = out_item_single[..., 5:]

        cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        w = w.reshape(-1)
        h = h.reshape(-1)
        cls_emb = cls_emb.reshape(-1, 80)
        conf = conf.reshape(-1)
        cls_argmax = cls_argmax.reshape(-1)

        x_top_left = x - w / 2.
        y_top_left = y - h / 2.
            # create all False
        flag = np.random.random(cls_emb.shape) > sys.maxsize
        is_person = np.zeros(cls_emb.shape[0])
        for i in range(flag.shape[0]):
            c = cls_argmax[i]
            flag[i, c] = True
            if c == 0:
                is_person[i] = 1
        confidence = cls_emb[flag] * conf * is_person

        for x_lefti, y_lefti, wi, hi, confi, _ in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
            if confi < threshold:
                continue
            x_lefti = max(0, x_lefti)
            y_lefti = max(0, y_lefti)
            wi = min(wi, ori_w)
            hi = min(hi, ori_h)

            bboxes.append([x_lefti, y_lefti, wi, hi, confi])
    return np.array(bboxes, dtype=np.float32)
