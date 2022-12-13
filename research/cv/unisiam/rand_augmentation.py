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
import math
import random
import numpy as np

import PIL
from PIL import Image, ImageOps, ImageEnhance


PIL_VERSION = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

MAX_AUG_LEVEL = 10.


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and PIL_VERSION < (5, 0): kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def auto_contrast(x, **__):
    return ImageOps.autocontrast(x)


def equalize(x, **__):
    return ImageOps.equalize(x)


def invert(x, **__):
    return ImageOps.invert(x)

def _rotate_pil_0(x, rot_degrees, **kwargs):
    return x.rotate(rot_degrees, **kwargs)


def _rotate_pil_1(x, rot_degrees, **kwargs):
    w, h = x.size
    post_trans = (0, 0)
    rotn_center = (w / 2.0, h / 2.0)
    angle = -math.radians(rot_degrees)
    aff_matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    aff_matrix[2], aff_matrix[5] = transform(
        -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], aff_matrix
    )
    aff_matrix[2] += rotn_center[0]
    aff_matrix[5] += rotn_center[1]
    return x.transform(x.size, Image.AFFINE, aff_matrix, **kwargs)


def _rotate_pil_2(x, rot_degrees, **kwargs):
    return x.rotate(rot_degrees, resample=kwargs['resample'])


def rotate(x, rot_degrees, **kwargs):
    _check_args_tf(kwargs)
    if PIL_VERSION >= (5, 2):
        rotate_func = _rotate_pil_0
    elif PIL_VERSION >= (5, 0):
        rotate_func = _rotate_pil_1
    else:
        rotate_func = _rotate_pil_2
    return rotate_func(x, rot_degrees, **kwargs)


def posterize(x, bits2keep, **__):
    if bits2keep >= 8: return x
    return ImageOps.posterize(x, bits2keep)


def solarize(x, thresh, **__):
    return ImageOps.solarize(x, thresh)


def solarize_add(x, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if x.mode in ("L", "RGB"):
        if x.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return x.point(lut)
    return x


def color(x, factor, **__):
    return ImageEnhance.Color(x).enhance(factor)


def contrast(x, factor, **__):
    return ImageEnhance.Contrast(x).enhance(factor)


def brightness(x, factor, **__):
    return ImageEnhance.Brightness(x).enhance(factor)


def sharpness(x, factor, **__):
    return ImageEnhance.Sharpness(x).enhance(factor)


def shear_x(x, factor, **kwargs):
    _check_args_tf(kwargs)
    return x.transform(x.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(x, factor, **kwargs):
    _check_args_tf(kwargs)
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(x, pct, **kwargs):
    pixels = pct * x.size[0]
    _check_args_tf(kwargs)
    return x.transform(x.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(x, pct, **kwargs):
    pixels = pct * x.size[1]
    _check_args_tf(kwargs)
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def _randomly_negate(v):
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level):
    return (_randomly_negate((level / MAX_AUG_LEVEL) * 30.),)


def _enhance_level_to_arg(level):
    return ((level / MAX_AUG_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
    return (_randomly_negate((level / MAX_AUG_LEVEL) * 0.3),)


def _translate_rel_level_to_arg(level):
    return (_randomly_negate((level / MAX_AUG_LEVEL) * 0.45),)


def _posterize_tpu_level_to_arg(level):
    return (int((level / MAX_AUG_LEVEL) * 4),)


def _solarize_level_to_arg(level):
    return (int((level / MAX_AUG_LEVEL) * 256),)


def _solarize_add_level_to_arg(level):
    return (int((level / MAX_AUG_LEVEL) * 110),)


def get_op(name):
    op_dict = {
        'AutoContrast': (auto_contrast, None),
        'Equalize': (equalize, None),
        'Invert': (invert, None),
        'Rotate': (rotate, _rotate_level_to_arg),
        'Posterize': (posterize, _posterize_tpu_level_to_arg),
        'Solarize': (solarize, _solarize_level_to_arg),
        'SolarizeAdd': (solarize_add, _solarize_add_level_to_arg),
        'Color': (color, _enhance_level_to_arg),
        'Contrast': (contrast, _enhance_level_to_arg),
        'Brightness': (brightness, _enhance_level_to_arg),
        'Sharpness': (sharpness, _enhance_level_to_arg),
        'ShearX': (shear_x, _shear_level_to_arg),
        'ShearY': (shear_y, _shear_level_to_arg),
        'TranslateXRel': (translate_x_rel, _translate_rel_level_to_arg),
        'TranslateYRel': (translate_y_rel, _translate_rel_level_to_arg),
    }
    return op_dict[name]


RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]

class RandAugmentationOp:

    def __init__(self, name, prob, magnitude, ra_params):
        self.aug, self.get_level = get_op(name)
        self.prob = prob
        self.magnitude = magnitude
        self.ra_params = ra_params.copy()
        self.kwargs = dict(
            fillcolor=ra_params['img_mean'],
            resample=(Image.BILINEAR, Image.BICUBIC),
        )
        self.magnitude_std = self.ra_params['magnitude_std']

    def __call__(self, x):
        if random.random() > self.prob: return x
        magnitude = random.gauss(self.magnitude, self.magnitude_std)
        magnitude = min(MAX_AUG_LEVEL, max(0, magnitude))
        level_args = self.get_level(magnitude, self.ra_params) if self.get_level is not None else tuple()
        return self.aug(x, *level_args, **self.kwargs)


def rand_augment_ops(magnitude, ra_params):
    return [RandAugmentationOp(
        name, prob=0.5, magnitude=magnitude, ra_params=ra_params) for name in RAND_TRANSFORMS]


class RandAugmentation:
    def __init__(self, used_ops, n_layer=2):
        self.used_ops = used_ops
        self.n_layer = n_layer

    def __call__(self, x):
        selected_ops = np.random.choice(self.used_ops, self.n_layer, replace=True, p=None)
        for op in selected_ops:
            x = op(x)
        return x


def rand_augmentation(n_layer, magnitude, ra_params):
    """
    Create a RandAugmentation transform
    """
    ra_ops = rand_augment_ops(magnitude=magnitude, ra_params=ra_params)
    return RandAugmentation(ra_ops, n_layer)
