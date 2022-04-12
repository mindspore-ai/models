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

"""Base dataset generator definition."""
import random

import cv2
import numpy as np
from mindspore import Tensor
from mindspore import ops as P
from mindspore.common import dtype


class BaseDataset:
    """Base dataset generator class."""
    def __init__(self,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=None,
                 std=None):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.files = []

    def __len__(self):
        """ Dataset length """
        return len(self.files)

    def input_transform(self, image):
        """Transform data format of images."""
        image = image.astype(np.float32)[:, :, ::-1]    # BGR2RGB
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        """Transform data format of labels."""
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, shape, padvalue):
        """Pad an image."""
        pad_image = image.copy()
        pad_h = max(shape[0] - h, 0)
        pad_w = max(shape[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        """Crop a feature at a random location."""
        h, w, _ = image.shape
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        """Augment feature into different scales."""
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w, _ = image.shape
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def gen_sample(self, image, label, multi_scale=False, is_flip=False):
        """Data preprocessing."""
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        image = self.input_transform(image)     # HWC
        label = self.label_transform(label)     # HW

        image = image.transpose((2, 0, 1))      # CHW

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, None,
                               fx=self.downsample_rate,
                               fy=self.downsample_rate,
                               interpolation=cv2.INTER_NEAREST)
        # image CHW, label HW
        return image, label

    def inference(self, model, image, flip=False):
        """Inference using one feature."""
        shape = image.shape
        pred = model(image)
        pred = pred[-1] # image NCHW
        pred = P.ResizeBilinear((shape[-2], shape[-1]))(pred)
        if flip:
            flip_img = image.asnumpy()[:, :, :, ::-1]
            flip_output = model(Tensor(flip_img.copy()))
            flip_output = flip_output[-1]
            flip_output = P.ResizeBilinear((shape[-2], shape[-1]))(flip_output)
            flip_pred = flip_output.asnumpy()
            flip_pred = Tensor(flip_pred[:, :, :, ::-1])
            pred = P.Add()(pred, flip_pred)
            pred = Tensor(pred.asnumpy() * 0.5)
        return P.Exp()(pred)

    def multi_scale_inference(self, model, image, scales=None, flip=False):
        """Inference using multi-scale features."""
        batch, _, ori_height, ori_width = image.shape
        assert batch == 1, "only supporting batchsize 1."
        image = image.asnumpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)

        final_pred = Tensor(np.zeros([1, self.num_classes, ori_height, ori_width]), dtype=dtype.float32)
        padvalue = -1.0 * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale, rand_crop=False)
            height, width = new_img.shape[:-1]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = Tensor(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width,
                                             self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = Tensor(np.zeros([1, self.num_classes, new_h, new_w]), dtype=dtype.float32)
                count = Tensor(np.zeros([1, 1, new_h, new_w]), dtype=dtype.float32)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img,
                                                      h1 - h0,
                                                      w1 - w0,
                                                      self.crop_size,
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = Tensor(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = P.ResizeBilinear((ori_height, ori_width))(preds)
            final_pred += preds
            final_pred = P.Add()(final_pred, preds)
        return final_pred
