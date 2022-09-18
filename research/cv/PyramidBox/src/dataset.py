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

import random
from PIL import Image
import numpy as np

from mindspore import dataset as ds

from src.augmentations import preprocess
from src.prior_box import PriorBox
from src.bbox_utils import match_ssd
from src.config import cfg


class WIDERDataset:
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDataset, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []
        prior_box = PriorBox(cfg)
        self.default_priors = prior_box.forward()
        self.num_priors = self.default_priors.shape[0]
        self.match = match_ssd
        self.threshold = cfg.FACE.OVERLAP_THRESH
        self.variance = cfg.VARIANCE

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if box:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, face_loc, face_conf, head_loc, head_conf = self.pull_item(index)
        return img, face_loc, face_conf, head_loc, head_conf

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(img, bbox_labels, self.mode)
            sample_labels = np.array(sample_labels)
            if sample_labels.size > 0:
                face_target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (face_target[:, 2] > face_target[:, 0]).any()
                assert (face_target[:, 3] > face_target[:, 1]).any()

                face_box = face_target[:, :-1]
                head_box = self.expand_bboxes(face_box)
                head_target = np.hstack((head_box, face_target[
                                        :, -1][:, np.newaxis]))
                break
            else:
                index = random.randrange(0, self.num_samples)

        face_truth = face_target[:, :-1]
        face_label = face_target[:, -1]

        face_loc_t, face_conf_t = self.match(self.threshold, face_truth, self.default_priors,
                                             self.variance, face_label)
        head_truth = head_target[:, :-1]
        head_label = head_target[:, -1]
        head_loc_t, head_conf_t = self.match(self.threshold, head_truth, self.default_priors,
                                             self.variance, head_label)
        return img, face_loc_t, face_conf_t, head_loc_t, head_conf_t


    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes

    def expand_bboxes(self,
                      bboxes,
                      expand_left=2.,
                      expand_up=2.,
                      expand_right=2.,
                      expand_down=2.):
        expand_bboxes = []
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            w = xmax - xmin
            h = ymax - ymin
            ex_xmin = max(xmin - w / expand_left, 0.)
            ex_ymin = max(ymin - h / expand_up, 0.)
            ex_xmax = max(xmax + w / expand_right, 0.)
            ex_ymax = max(ymax + h / expand_down, 0.)
            expand_bboxes.append([ex_xmin, ex_ymin, ex_xmax, ex_ymax])
        expand_bboxes = np.array(expand_bboxes)
        return expand_bboxes

def create_val_dataset(mindrecord_file, batch_size, device_num=1, device_id=0, num_workers=8):
    """
    Create user-defined mindspore dataset for training
    """
    column_names = ['img', 'face_loc', 'face_conf', 'head_loc', 'head_conf']
    ds.config.set_num_parallel_workers(num_workers)
    ds.config.set_enable_shared_mem(False)
    ds.config.set_prefetch_size(batch_size * 2)

    train_dataset = ds.MindDataset(mindrecord_file, columns_list=column_names, shuffle=True,
                                   shard_id=device_id, num_shards=device_num)
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)

    return train_dataset

def create_train_dataset(cfg_, batch_size, device_num=1, device_id=0, num_workers=8):
    """
    Create user-defined mindspore dataset for training
    """
    column_names = ['img', 'face_loc', 'face_conf', 'head_loc', 'head_conf']
    ds.config.set_num_parallel_workers(num_workers)
    ds.config.set_enable_shared_mem(False)
    ds.config.set_prefetch_size(batch_size * 2)
    train_dataset = ds.GeneratorDataset(WIDERDataset(cfg_.FACE.TRAIN_FILE, mode='train'),
                                        column_names=column_names, shuffle=True, num_shards=device_num,
                                        shard_id=device_id)
    train_dataset = train_dataset.batch(batch_size=batch_size)

    return train_dataset
