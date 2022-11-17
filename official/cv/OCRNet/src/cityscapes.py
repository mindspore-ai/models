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

"""Dataset Cityscapes generator."""
import cv2
import numpy as np

import mindspore.ops as P
from mindspore import Tensor
from mindspore.common import dtype

from src.basedataset import BaseDataset


class Cityscapes(BaseDataset):
    """Dataset Cityscapes generator."""
    def __init__(self,
                 root,
                 num_samples=None,
                 num_classes=19,
                 multi_scale=False,
                 flip=False,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=None,
                 std=None,
                 is_train=True):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std)

        self._index = 0
        self.root = root
        if is_train:
            self.list_path = root + "/train.lst"
        else:
            self.list_path = root + "/val.lst"
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        list_file = open(self.list_path)
        img_list = [line.strip().split() for line in list_file]
        list_file.close()
        self.img_list = [(self.root+"/"+vector[0], self.root+"/"+vector[1]) for vector in img_list]
        self._number = len(self.img_list)

        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}
        self.class_weights = Tensor([0.8373, 0.918, 0.866, 1.0345,
                                     1.0166, 0.9969, 0.9754, 1.0489,
                                     0.8786, 1.0023, 0.9539, 0.9843,
                                     1.1116, 0.9037, 1.0865, 1.0955,
                                     1.0865, 1.1529, 1.0507], dtype=dtype.float32)

    def __len__(self):
        return self._number

    def __getitem__(self, index):
        if index < self._number:
            image_path = self.img_list[index][0]
            label_path = self.img_list[index][1]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label)
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        else:
            raise StopIteration
        return image.copy(), label.copy()

    def show(self):
        """Show the total number of val data."""
        print("Total number of data vectors: ", self._number)
        for line in self.img_list:
            print(line)

    def convert_label(self, label, inverse=False):
        """Convert classification ids in labels."""
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def multi_scale_inference(self, model, image, scales=None, flip=False):
        """Inference using multi-scale features from dataset Cityscapes."""
        batch, _, ori_height, ori_width = image.shape
        assert batch == 1, "only supporting batchsize 1."
        image = image.asnumpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)

        final_pred = Tensor(np.zeros([1, self.num_classes, ori_height, ori_width]), dtype=dtype.float32)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = Tensor(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds.asnumpy()
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = np.zeros([1, self.num_classes, new_h, new_w]).astype(np.float32)

                count = np.zeros([1, 1, new_h, new_w]).astype(np.float32)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = Tensor(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred.asnumpy()[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = Tensor(preds)
            preds = P.ResizeBilinear((ori_height, ori_width))(preds)
            final_pred = P.Add()(final_pred, preds)
        return final_pred
