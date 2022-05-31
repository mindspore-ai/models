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
"""Data Augmentation"""

import random
import numpy as np
import PIL
import mindspore.dataset.vision as vision

RESAMPLE_MODE = PIL.Image.BICUBIC

RANDOM_MIRROR = True

def ShearX(img, v, resample=RESAMPLE_MODE):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0),
                         resample=resample)

def ShearY(img, v, resample=RESAMPLE_MODE):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0),
                         resample=resample)

def TranslateX(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         resample=resample)

def TranslateY(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         resample=resample)

def TranslateXabs(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert v >= 0
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         resample=resample)

def TranslateYabs(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert v >= 0
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         resample=resample)

def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = PIL.Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def Posterize(img, v):  # [4, 8]
    #assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    """Cutout absolute value"""
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    """Cutout"""
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)

def rand_augment_list():  # 16 oeprations and their ranges
    """Random augment list"""
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l

class RandAugment:
    """RandAugment class"""
    def __init__(self, n, m, from_pil=False, as_pil=False):
        self.n = n
        self.m = m
        self.augment_list = rand_augment_list()
        self.to_pil = vision.ToPIL()
        self.to_tensor = vision.ToTensor()
        self.from_pil = from_pil
        self.as_pil = as_pil

    def __call__(self, img):
        if not self.from_pil:
            img = self.to_pil(img)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > random.uniform(0.2, 0.8):
                continue
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        if not self.as_pil:
            img = self.to_tensor(img)
        return img
