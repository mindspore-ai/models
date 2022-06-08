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
"""base dataset"""

from PIL import Image
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.transforms import Compose
from mindspore.dataset.vision import Inter


class BaseDataset:
    """BaseDataset"""
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 0

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser


def get_transform(opt):
    """get_transform"""
    # pylint: disable=W0108
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.fineSize]
        transform_list.append(vision.Resize(osize, Inter.BICUBIC))  # PIL
        transform_list.append(vision.RandomCrop(opt.fineSize))  # PIL
    elif opt.resize_or_crop == 'crop':
        transform_list.append(vision.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(
            lambda img: __scale_width(img, opt.fineSize))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(
            lambda img: __scale_width(img, opt.loadSize))
        transform_list.append(vision.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none':
        transform_list.append(
            lambda img: __adjust(img))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(vision.RandomHorizontalFlip())

    transform_list += [vision.ToTensor(),
                       vision.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5), is_hwc=False)]
    return Compose(transform_list)

# just modify the width and height to be multiple of 4
# img: PIL
def __adjust(img):
    """__adjust"""
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


# img: PIL
def __scale_width(img, target_width):
    """__scale_width"""
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if ow == target_width and oh % mult == 0:
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    """__print_size_warning"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
