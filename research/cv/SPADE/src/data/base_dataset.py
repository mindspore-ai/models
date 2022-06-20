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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" BaseDataset"""

import random
import numpy as np
from PIL import Image

class BaseDataset:
    def __init__(self):
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, img, params, method=Image.BICUBIC):
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        img = __scale(img, osize, method)
        # transform_list.append(c_vision.Resize(osize, interpolation=Inter.BICUBIC))
    elif 'scale_width' in opt.preprocess_mode:
        img = __scale_width(img, opt.load_size, method)
    elif 'scale_shortside' in opt.preprocess_mode:
        img = __scale_shortside(img, opt.load_size, method)

    if 'crop' in opt.preprocess_mode:
        img = __crop(img, params['crop_pos'], opt.crop_size)

    if opt.preprocess_mode == 'none':
        base = 32
        img = __make_power_2(img, base, method)

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        img = __resize(img, w, h, method)

    if opt.isTrain and not opt.no_flip:
        img = __flip(img, params['flip'])

    return img

def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)

def __make_by_255(img):
    return img * 255

def __set_255(img, val):
    img[img == 255] = val
    return img

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale(img, osize, method=Image.BICUBIC):
    return img.resize((osize[0], osize[1]), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if ss == target_width:
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
