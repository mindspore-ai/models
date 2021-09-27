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
create train or eval dataset.
"""
import os
import pickle

import cv2
import numpy as np

from src.common.logger import Logger
from src.paths import Data
from src.utils.util import pre_process


class FinetuneAndSVMDataset:
    """
    FinetuneAndSVMDataset
    """

    def __init__(self, finetune_or_svm, phases=None):
        if phases is None:
            phases = ['train', 'val']
        data_root = Data.finetune if finetune_or_svm else Data.svm
        self.data_root = data_root
        self.jpeg_root_path = Data.jpeg
        self.pos_rects = list()
        self.neg_rects = list()
        self.pos_labels = list()
        self.pos_key_list = list()
        self.neg_key_list = list()
        self.mean = [0.453 * 255, 0.433 * 255, 0.398 * 255]
        self.std = [0.270 * 255, 0.268 * 255, 0.279 * 255]
        Logger().info("fintune/svm dataset initializes phases: %s" % phases)
        Logger().info("data directory: %s" % self.data_root)
        for phase in phases:
            neg_rects, neg_key_list, pos_rects, pos_labels, pos_key_list = self.load_data(phase=phase)
            self.neg_rects.extend(neg_rects)
            self.neg_key_list.extend(neg_key_list)
            self.pos_rects.extend(pos_rects)
            self.pos_labels.extend(pos_labels)
            self.pos_key_list.extend(pos_key_list)
        pos_num = len(self.pos_labels)
        neg_num = len(self.neg_rects)
        Logger().info("Total samples:%d positive samples:%d negative samples：%d"
                      % (pos_num + neg_num, pos_num, neg_num))

    def load_data(self, phase):
        """

        :param phase: phase
        :return: data
        """
        neg_rects, neg_key_list, pos_rects, pos_labels, pos_key_list = [], [], [], [], []
        pickle_dir = os.path.join(self.data_root, phase)
        pickle_list = os.listdir(pickle_dir)
        for pickle_filename in pickle_list:
            finetune_data = pickle.load(open(os.path.join(pickle_dir, pickle_filename), 'rb'))
            for key, value in finetune_data.items():
                for i_value in value:
                    fine_label = i_value[-1]
                    fine_rect = i_value[0:-1]
                    if fine_label == 20:
                        neg_rects.append(fine_rect)
                        neg_key_list.append(key)
                    else:
                        pos_rects.append(fine_rect)
                        pos_labels.append(fine_label)
                        pos_key_list.append(key)

        return neg_rects, neg_key_list, pos_rects, pos_labels, pos_key_list

    def __getitem__(self, index: int):
        rand_pos = np.random.rand(256)
        pos_imgs = list()
        pos_labels = list()
        for i in range(256):
            pos_img_index = int(len(self.pos_rects) * rand_pos[i]) - 1
            pos_img_id = self.pos_key_list[pos_img_index]
            pos_img = cv2.imread(os.path.join(self.jpeg_root_path, pos_img_id + '.jpg'))
            rect = self.pos_rects[pos_img_index]
            xmin, ymin, xmax, ymax = rect
            crop_pos_img = pos_img[ymin:ymax, xmin:xmax]
            crop_pos_img = pre_process(crop_pos_img, self.mean, self.std)
            pos_imgs.append(crop_pos_img)
            pos_labels.append(self.pos_labels[pos_img_index])

        rand_neg = np.random.rand(256)
        neg_imgs = list()
        neg_labels = list()
        for i in range(256):
            neg_img_index = int(len(self.neg_rects) * rand_neg[i]) - 1
            neg_img_id = self.neg_key_list[neg_img_index]
            neg_img = cv2.imread(os.path.join(self.jpeg_root_path, neg_img_id + '.jpg'))
            rect = self.neg_rects[neg_img_index]
            xmin, ymin, xmax, ymax = rect
            crop_neg_img = neg_img[ymin:ymax, xmin:xmax]
            crop_neg_img = pre_process(crop_neg_img, self.mean, self.std)
            neg_imgs.append(crop_neg_img)
            neg_labels.append(20)
        samplers = pos_imgs + neg_imgs
        samplers = np.array(samplers, dtype=np.float32)
        labels = pos_labels + neg_labels
        labels = np.array(labels, dtype=np.int32)
        return samplers, labels

    def __len__(self) -> int:
        return int(len(self.pos_labels) / 256)


class FinetuneAndSVMDataset_test:
    """
    FinetuneAndSVMDataset_test
    """

    def __init__(self, finetune_or_svm, phases=None):
        if phases is None:
            phases = ['test']
        data_root = Data.finetune if finetune_or_svm else Data.svm
        self.data_root = data_root
        self.jpeg_root_path = Data.jpeg_test
        self.pos_rects = list()
        self.neg_rects = list()
        self.pos_labels = list()
        self.pos_key_list = list()
        self.neg_key_list = list()
        self.mean = [0.453 * 255, 0.433 * 255, 0.398 * 255]
        self.std = [0.270 * 255, 0.268 * 255, 0.279 * 255]
        Logger().info("fintune/svm dataset initializes phases: %s" % phases)
        Logger().info("data directory: %s" % self.data_root)
        for phase in phases:
            neg_rects, neg_key_list, pos_rects, pos_labels, pos_key_list = self.load_data(phase=phase)
            self.neg_rects.extend(neg_rects)
            self.neg_key_list.extend(neg_key_list)
            self.pos_rects.extend(pos_rects)
            self.pos_labels.extend(pos_labels)
            self.pos_key_list.extend(pos_key_list)
        pos_num = len(self.pos_labels)
        neg_num = len(self.neg_rects)
        Logger().info("Total samples:%d positive samples:%d negative samples：%d"
                      % (pos_num + neg_num, pos_num, neg_num))

    def load_data(self, phase):
        """

        :param phase: phase
        :return: data
        """
        neg_rects, neg_key_list, pos_rects, pos_labels, pos_key_list = [], [], [], [], []
        pickle_dir = os.path.join(self.data_root, phase)
        pickle_list = os.listdir(pickle_dir)
        for pickle_filename in pickle_list:
            finetune_data = pickle.load(open(os.path.join(pickle_dir, pickle_filename), 'rb'))
            for key, value in finetune_data.items():
                for i_value in value:
                    fine_label = i_value[-1]
                    fine_rect = i_value[0:-1]
                    if fine_label == 20:
                        neg_rects.append(fine_rect)
                        neg_key_list.append(key)
                    else:
                        pos_rects.append(fine_rect)
                        pos_labels.append(fine_label)
                        pos_key_list.append(key)

        return neg_rects, neg_key_list, pos_rects, pos_labels, pos_key_list

    def __getitem__(self, index: int):
        pos_imgs = list()
        pos_labels = list()
        index = index * 256
        for pos_img_index in range(index, index + 256):
            pos_img_id = self.pos_key_list[pos_img_index]
            pos_img = cv2.imread(os.path.join(self.jpeg_root_path, pos_img_id + '.jpg'))
            rect = self.pos_rects[pos_img_index]
            xmin, ymin, xmax, ymax = rect
            crop_pos_img = pos_img[ymin:ymax, xmin:xmax]
            crop_pos_img = pre_process(crop_pos_img, self.mean, self.std)
            pos_imgs.append(crop_pos_img)
            pos_labels.append(self.pos_labels[pos_img_index])

        rand_neg = np.random.rand(256)
        neg_imgs = list()
        neg_labels = list()
        for i in range(256):
            neg_img_index = int(len(self.neg_rects) * rand_neg[i]) - 1
            neg_img_id = self.neg_key_list[neg_img_index]
            neg_img = cv2.imread(os.path.join(self.jpeg_root_path, neg_img_id + '.jpg'))
            rect = self.neg_rects[neg_img_index]
            xmin, ymin, xmax, ymax = rect
            crop_neg_img = neg_img[ymin:ymax, xmin:xmax]
            crop_neg_img = pre_process(crop_neg_img, self.mean, self.std)
            neg_imgs.append(crop_neg_img)
            neg_labels.append(20)
        samplers = pos_imgs + neg_imgs
        samplers = np.array(samplers, dtype=np.float32)
        labels = pos_labels + neg_labels
        labels = np.array(labels, dtype=np.int32)
        return samplers, labels

    def __len__(self) -> int:
        return int(len(self.pos_labels) / 256)


class RegressionDataset:
    """
    RegressionDataset
    """

    def __init__(self, phases=None):
        if phases is None:
            phases = ['train', 'val']
        if "test" in phases:
            if len(phases) > 1:
                Logger().critical('phases length should be 1 while it contains "test"')
            self.jpeg_root_path = Data.jpeg_test
        else:
            self.jpeg_root_path = Data.jpeg
        self.reg_pos_rects = list()
        self.reg_pos_trans = list()
        self.reg_pos_keys = list()
        self.reg_cls = list()
        self.mean = [0.453 * 255, 0.433 * 255, 0.398 * 255]
        self.std = [0.270 * 255, 0.268 * 255, 0.279 * 255]
        Logger().info("regression dataset initializes phases: %s" % phases)

        for phase in phases:
            rects, trans_, keys, cls_ = self.load_data(phase=phase)
            self.reg_pos_rects.extend(rects)
            self.reg_pos_trans.extend(trans_)
            self.reg_pos_keys.extend(keys)
            self.reg_cls.extend(cls_)

        Logger().info('Dataset loading completed, number of samples： %s' % len(self.reg_cls))

    def load_data(self, phase):
        """

        :param phase: phase
        :return: data
        """
        rects, trans_, keys, cls_ = [], [], [], []
        pickle_dir = os.path.join(Data.regression, phase)
        pickle_list = os.listdir(pickle_dir)
        for pickle_filename in pickle_list:
            reg_data = pickle.load(open(os.path.join(pickle_dir, pickle_filename), 'rb'))
            for key, value in reg_data.items():
                for i_value in value:
                    reg_trans = i_value[5]
                    reg_rect = i_value[0:4]
                    reg_cls = i_value[4]
                    rects.append(reg_rect)
                    trans_.append(reg_trans)
                    keys.append(key)
                    cls_.append(reg_cls)

        return rects, trans_, keys, cls_

    def __getitem__(self, index: int):
        img_id = self.reg_pos_keys[index]
        img_ = cv2.imread(os.path.join(self.jpeg_root_path, img_id + '.jpg'))
        rect = self.reg_pos_rects[index]
        trans_ = self.reg_pos_trans[index]
        cls_ = self.reg_cls[index]
        x, y, w, h = rect
        xmin = x - w // 2
        xmax = x + w // 2
        ymin = y - h // 2
        ymax = y + h // 2
        crop_img = img_[ymin:ymax, xmin:xmax]
        crop_img = pre_process(crop_img, self.mean, self.std)
        crop_img = np.array(crop_img, dtype=np.float32)
        trans_ = np.array(trans_, dtype=np.float32)
        cls_onehot = np.zeros(20, dtype=np.float32)
        cls_onehot[cls_] = 1
        return crop_img, trans_, cls_onehot

    def __len__(self) -> int:
        return len(self.reg_pos_rects)
