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
"""data_loader"""
import copy
import random
import os.path as osp
from collections import defaultdict
import cv2
import numpy as np

def read_image_color(img_path):
    """
    Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            return None
    return img

def read_image_grayscale(img_path):
    """
    Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            return None
    return img

def get_eval_data(num_instances, batch_size, dataset):
    """
    get eval dataset
    """
    index_dic = defaultdict(list)
    for index, (_, vid, _, _, _, _, _, _) in enumerate(dataset):
        index_dic[vid].append(index)
    vids = list(index_dic.keys())

    batch_idxs_dict = defaultdict(list)
    for vid in vids:
        idxs = copy.deepcopy(index_dic[vid])
        if len(idxs) < num_instances:
            idxs = np.random.choice(idxs, size=num_instances, replace=True)
        random.shuffle(idxs)
        batch_idxs = []
        for idx in idxs:
            batch_idxs.append(idx)
            if len(batch_idxs) == num_instances:
                batch_idxs_dict[vid].append(batch_idxs)
                batch_idxs = []

    avai_vids = copy.deepcopy(vids)
    avai_vids_length = len(avai_vids)
    final_idxs = []
    num_vids_per_batch = batch_size // num_instances

    while avai_vids_length >= num_vids_per_batch:
        selected_vids = random.sample(avai_vids, num_vids_per_batch)
        for vid in selected_vids:
            batch_idxs = batch_idxs_dict[vid].pop(0)
            final_idxs.extend(batch_idxs)
            batch_idxs_dict_length = len(batch_idxs_dict[vid])
            if batch_idxs_dict_length == 0:
                avai_vids.remove(vid)
        avai_vids_length = len(avai_vids)

    return final_idxs

class ImageDataset():
    """Image ReID Dataset"""
    def __init__(self, dataset, batch_size, num_instances,
                 keyptaware=True, heatmapaware=True, segmentaware=True,
                 transform=None, imagesize=None, is_train=False):
        self.dataset = dataset
        self.keyptaware = keyptaware
        self.heatmapaware = heatmapaware
        self.segmentaware = segmentaware
        self.transform = transform
        self.imagesize = imagesize
        self.is_train = is_train

        self.segments = [(5, 15, 16, 17), (5, 6, 12, 15), (6, 10, 11, 12),
                         (23, 33, 34, 35), (23, 24, 30, 33), (24, 28, 29, 30),
                         (10, 11, 29, 28), (11, 12, 30, 29), (12, 13, 31, 30),
                         (13, 14, 32, 31), (14, 15, 33, 32), (15, 16, 34, 33),
                         (16, 17, 35, 34)]

        self.conf_thld = 0.5

        if self.is_train:
            self.final_idxs = get_eval_data(num_instances, batch_size, self.dataset)
            self.length = len(self.final_idxs)
        else:
            self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_chnls = []

        if self.is_train:
            _id = self.final_idxs[index]
            img_path, vid, camid, vcolor, vtype, vkeypt, heatmap_dir_path, segment_dir_path = self.dataset[_id]
        else:
            img_path, vid, camid, vcolor, vtype, vkeypt, heatmap_dir_path, segment_dir_path = self.dataset[index]

        img_orig = read_image_color(img_path)
        height_orig, width_orig, _ = img_orig.shape
        img_b, img_g, img_r = cv2.split(img_orig)
        img_chnls.extend([img_r, img_g, img_b])

        if self.heatmapaware:
            for h in range(36):
                heatmap_path = osp.join(heatmap_dir_path, "%02d.jpg" % h)
                heatmap = read_image_grayscale(heatmap_path)
                heatmap = cv2.resize(heatmap, dsize=(width_orig, height_orig))
                img_chnls.append(heatmap)

        if self.segmentaware:
            for s in range(len(self.segments)):
                segment_flag = True
                for k in self.segments[s]:
                    if vkeypt[k * 3+2] < self.conf_thld:
                        segment_flag = False
                        break
                if segment_flag:
                    segment_path = osp.join(segment_dir_path, "%02d.jpg" % s)
                    segment = read_image_grayscale(segment_path)
                    segment = cv2.resize(segment, dsize=(width_orig, height_orig))
                else:
                    segment = np.zeros((height_orig, width_orig), np.uint8)
                img_chnls.append(segment)

        img = np.stack(img_chnls, axis=2)
        img = self.transform(img, vkeypt)
        vkeypt = np.asarray(vkeypt)

        if self.keyptaware:
            for k in range(vkeypt.size):
                if k % 3 == 0:
                    vkeypt[k] = (vkeypt[k] / float(self.imagesize[0])) - 0.5
                elif k % 3 == 1:
                    vkeypt[k] = (vkeypt[k] / float(self.imagesize[1])) - 0.5
                elif k % 3 == 2:
                    vkeypt[k] -= 0.5

        return img, vid, camid, vcolor, vtype, vkeypt
