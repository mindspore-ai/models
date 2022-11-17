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
# pytorch dataset for SL learning
# matlab code (line 26-33):
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference:
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py


import numpy as np

from src.utils.gen_samples import gen_samples
from src.utils.overlap_ratio import overlap_ratio
from src.utils.gen_action_labels import gen_action_labels
from src.utils.augmentations import ADNet_Augmentation


class OnlineAdaptationDatasetStorage():
    def __init__(self, initial_frame, first_box, opts, args, positive=True):
        self.opts = opts
        self.positive = positive

        if positive:
            self.max_num_past_frames = opts['nFrames_long']
        else:
            self.max_num_past_frames = opts['nFrames_short']

        self.transform = ADNet_Augmentation(opts)

        self.train_db = []

        self.add_frame_then_generate_samples(initial_frame, first_box)

    def get_item(self, index):  # __getitem__
        # find out which train_db's index is the index
        remaining_idx = index

        train_db_idx = 0
        for train_db_idx, train_db_ in enumerate(self.train_db):
            if remaining_idx < len(train_db_['bboxes']):
                break
            remaining_idx -= len(train_db_['bboxes'])

        # get the data
        im = self.train_db[train_db_idx]['past_frame']
        bbox = self.train_db[train_db_idx]['bboxes'][remaining_idx]
        action_label = np.array(self.train_db[train_db_idx]['labels'][remaining_idx], dtype=np.float32)
        score_label = self.train_db[train_db_idx]['score_labels'][remaining_idx]

        if self.transform is not None:
            im, bbox, action_label, score_label = self.transform(im, bbox, action_label, score_label)
        return im, bbox, action_label, score_label

    def get_len(self):  # __len__
        number_samples = 0
        for train_db_ in self.train_db:
            number_samples += len(train_db_['bboxes'])
        return number_samples

    # add past frame...
    def add_frame_then_generate_samples(self, frame, curr_box):
        init = not self.train_db

        train_db_ = {
            'past_frame': frame,
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }

        self.opts['imgSize'] = frame.shape[:2]

        bboxes, labels, score_labels = self.generate_samples(curr_box, positive=self.positive, init=init)
        train_db_['bboxes'] = bboxes
        train_db_['labels'] = labels
        train_db_['score_labels'] = score_labels

        self.train_db.append(train_db_)

        # delete old frames if the history is full
        while len(self.train_db) > self.max_num_past_frames:  # saver with while instead of if
            del self.train_db[0]

    # generate samples from past frames, called if tracking success...
    # generate pos/neg samples
    # private class
    def generate_samples(self, curr_bbox, positive, init=False):
        if init:
            if positive:
                n = self.opts['nPos_init']
                thre = self.opts['posThre_init']
            else:
                n = self.opts['nNeg_init']
                thre = self.opts['negThre_init']
        else:
            if positive:
                n = self.opts['nPos_online']
                thre = self.opts['posThre_online']
            else:
                n = self.opts['nNeg_online']
                thre = self.opts['negThre_online']

        assert n > 0, "if n = 0, don't initialize this class"

        if positive:
            examples = gen_samples('gaussian', curr_bbox, n * 2, self.opts,
                                   self.opts['finetune_trans'], self.opts['finetune_scale_factor'])
            r = overlap_ratio(examples, np.tile(curr_bbox, (len(examples), 1)))
            examples = examples[np.array(r) > thre]
            examples = examples[np.random.randint(low=0, high=len(examples),
                                                  size=min(len(examples), n)), :]

            action_labels = gen_action_labels(self.opts['num_actions'], self.opts, np.array(examples),
                                              curr_bbox)
            # score labels: 1 is positive. 0 is negative
            score_labels = list(np.ones(len(examples), dtype=int))

        else:
            examples = gen_samples('uniform', curr_bbox, n * 2, self.opts, 2, 5)
            r = overlap_ratio(examples, np.tile(curr_bbox, (len(examples), 1)))
            examples = examples[np.array(r) < thre]
            examples = examples[np.random.randint(low=0, high=len(examples),
                                                  size=min(len(examples), n)), :]

            action_labels = np.full((self.opts['num_actions'], len(examples)), fill_value=-1)
            # score labels: 1 is positive. 0 is negative
            score_labels = list(np.zeros(len(examples), dtype=int))

        action_labels = np.transpose(action_labels).tolist()
        bboxes = examples
        labels = action_labels

        return bboxes, labels, score_labels


# should be initialized again whenever the dataset_storage has changed
class OnlineAdaptationDataset:
    def __init__(self, dataset_storage):
        self.dataset_storage = dataset_storage

    def __getitem__(self, index):
        return self.dataset_storage.get_item(index)

    def __len__(self):
        return self.dataset_storage.get_len()
