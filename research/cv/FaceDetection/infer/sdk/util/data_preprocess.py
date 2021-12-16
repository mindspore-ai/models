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
"""Face detection yolov3 data pre-process."""
import io
import numpy as np
from PIL import Image, ImageOps


def hwc_to_chw(img):
    """
    Transpose the input image; shape (H, W, C) to shape (C, H, W).

    Args:
        img (numpy.ndarray): Image to be converted.

    Returns:
        img (numpy.ndarray), Converted image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be NumPy array. Got {}.'.format(type(img)))
    if img.ndim != 3:
        raise TypeError('img dimension should be 3. Got {}.'.format(img.ndim))
    return img.transpose(2, 0, 1).copy()


def to_tensor(img, output_type=np.float32):
    """
        Change the input image (PIL image or NumPy image array) to NumPy format.

        Args:
            img (Union[PIL image, numpy.ndarray]): Image to be converted.
            output_type: The datatype of the NumPy output. e.g. np.float32

        Returns:
            img (numpy.ndarray), Converted image.
        """
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise TypeError("img dimension should be 2 or 3. Got {}.".format(img.ndim))

    if img.ndim == 2:
        img = img[:, :, None]

    img = hwc_to_chw(img)
    img = img / 255.
    return img.astype(output_type)


class SingleScaleTrans:
    """
        SingleScaleTrans
    """

    def __init__(self, resize=None, max_anno_count=200):
        if resize is None:
            resize = [768, 448]
        self.resize = (resize[0], resize[1])
        self.max_anno_count = max_anno_count

    def __call__(self, image, ann, image_names, image_size):
        size = self.resize
        resize_letter_box_op = ResizeLetterbox(input_dim=size)

        ret_imgs = []
        ret_anno = []

        data = io.BytesIO(image)
        image = Image.open(data)
        img_pil = image.convert('RGB')
        input_data = img_pil, ann
        input_data = resize_letter_box_op(*input_data)
        image_arr = to_tensor(input_data[0])
        ret_imgs.append(image_arr)
        ret_anno.append(input_data[1])

        self.deal_anno(ret_anno)

        return np.array(ret_imgs), np.array(ret_anno), np.array([image_names]), image_size

    def deal_ann_size_name_to_array(self, ann, image_names, image_size):
        ret_anno = [np.asarray(ann)]
        self.deal_anno(ret_anno)
        return np.array(ret_anno), np.array([image_names]), image_size

    def deal_anno(self, ret_anno):
        for i, anno in enumerate(ret_anno):
            anno_count = anno.shape[0]
            if anno_count < self.max_anno_count:
                ret_anno[i] = np.concatenate(
                    (ret_anno[i], np.zeros((self.max_anno_count - anno_count, 6), dtype='float32')), axis=0)
            else:
                ret_anno[i] = ret_anno[i][:self.max_anno_count]


class ResizeLetterbox:
    """ Resize the image to input_dim.

    Args:
        input_dim: Input size of network.
    """

    def __init__(self, fill_color=127, input_dim=(1408, 768)):
        self.fill_color = fill_color
        self.crop_info = None
        self.output_w = None
        self.output_h = None
        self.input_dim = input_dim
        self.pad = None
        self.scale = None

    def __call__(self, img, annos):
        if img is None:
            return None, None
        if isinstance(img, Image.Image):
            img = self._tf_pil(img)

        annos = np.asarray(annos)

        return img, annos

    def _tf_pil(self, img):
        """ Letterbox an image to fit in the network """

        net_w, net_h = self.input_dim

        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            resample_mode = Image.NEAREST
            img = img.resize((int(self.scale * im_w), int(self.scale * im_h)), resample_mode)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,) * channels)

        return img
