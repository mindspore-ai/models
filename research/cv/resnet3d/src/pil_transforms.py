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
"""
transforms by PIL.
"""
import numpy as np

import mindspore.dataset.vision as vision


class PILTrans:
    """
    Transforms for training.
    """

    def __init__(self, opt, mean, std):
        super(PILTrans).__init__()
        self.to_pil = vision.ToPIL()
        self.random_resized_crop = \
            vision.RandomResizedCrop(opt.sample_size, scale=(opt.train_crop_min_scale, 1.0),
                                     ratio=(opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio))
        self.random_horizontal_flip = vision.RandomHorizontalFlip(prob=0.5)
        self.color = vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)
        self.normalize = vision.Normalize(mean=mean, std=std)
        self.to_tensor = vision.ToTensor()
        self.resize = vision.Resize(opt.sample_size)
        self.center_crop = vision.CenterCrop(opt.sample_size)
        self.opt = opt

    def __call__(self, data, labels, batchInfo):
        data_ret = []
        for _, imgs in enumerate(data):
            imgs_ret = []
            for idx in range(0, 16):
                img = imgs[idx]
                img_pil = self.to_pil(img)
                if self.opt.train_crop == 'random':
                    img_pil = self.random_resized_crop(img_pil)
                else:
                    img_pil = self.resize(img_pil)
                    img_pil = self.center_crop(img_pil)
                if self.opt.h_flip:
                    img_pil = self.random_horizontal_flip(img_pil)
                if self.opt.colorjitter:
                    img_pil = self.color(img_pil)
                img_array = self.to_tensor(img_pil)
                img_array = self.normalize(img_array)
                imgs_ret.append(img_array)
            imgs_ret = np.array(imgs_ret)
            imgs_ret = imgs_ret.transpose((1, 0, 2, 3))  # DCHW -> CDHW
            data_ret.append(imgs_ret)

        return data_ret, labels


class EvalPILTrans:
    """
    Transforms for evaling.
    """

    def __init__(self, opt, mean, std):
        super(EvalPILTrans).__init__()
        self.to_pil = vision.ToPIL()
        self.resize = vision.Resize(opt.sample_size)
        self.center_crop = vision.CenterCrop(opt.sample_size)
        self.normalize = vision.Normalize(mean=mean, std=std)
        self.to_tensor = vision.ToTensor()

    def __call__(self, data, labels, batchInfo):
        data = data[0]
        N = data.shape[0]
        D = data.shape[1]
        data_tmp = []
        data_ret = []
        for i in range(0, N):
            video_ret = []
            video = data[i]
            for j in range(0, D):
                img = video[j]
                img_pil = self.to_pil(img)
                img_pil = self.resize(img_pil)
                img_pil = self.center_crop(img_pil)
                img_array = self.to_tensor(img_pil)
                img_array = self.normalize(img_array)
                video_ret.append(img_array)
            video_ret = np.array(video_ret)
            video_ret = video_ret.transpose(1, 0, 2, 3)
            data_tmp.append(video_ret)
        data_ret.append(data_tmp)
        return data_ret, labels
