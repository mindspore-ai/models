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

import json
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import mindspore.dataset as ds
from src.utils import ioa_with_anchors, iou_with_anchors

log = logging.getLogger(__name__)


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data

class VideoDataset():
    def __init__(self, cfg, subset="train"):
        self.temporal_scale = cfg.model.temporal_scale  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = cfg.mode
        self.feature_path = cfg.data.feature_path
        self.video_info_path = cfg.data.video_info
        self.video_anno_path = cfg.data.video_anno
        self._getDatasetDict()
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = sorted(list(self.video_dict.keys()))
        log.info("%s subset video numbers: %d", self.subset, len(self.video_list))

    def __getitem__(self, index):
        video_data = self.load_file(index)
        if self.subset == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data, confidence_score, match_score_start, match_score_end
        return (video_data,)

    def load_file(self, index):
        video_name = self.video_list[index]
        video_feat = np.load(self.feature_path + video_name + ".npy")
        video_feat = video_feat[:1000, :]
        video_feat = np.transpose(video_feat, (1, 0))
        video_feat = video_feat.astype("float32")
        return video_feat

    def get_dataset_dict(self):
        return self.video_dict

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_len_small = 3 * self.temporal_gap
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = np.array(gt_iou_map)

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = np.array(match_score_start)
        match_score_end = np.array(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)

def createDataset(cfg, mode) -> Tuple[ds.BatchDataset, dict]:
    """create BMN dataset"""

    subset = mode
    if mode == "eval":
        subset = 'validation'
    data = VideoDataset(cfg, subset)

    if mode == 'train':
        columns = ['features', 'confidence_score', 'match_score_start', 'match_score_end']
    else:
        columns = ['features']
    shuffle = mode == "train"
    drop_remainder = mode == "train"
    dataset = ds.GeneratorDataset(data, column_names=columns,
                                  shuffle=shuffle, num_parallel_workers=cfg.data.threads, max_rowsize=32)

    dataset = dataset.batch(cfg[mode].batch_size, drop_remainder=drop_remainder)
    return dataset, data.get_dataset_dict()
