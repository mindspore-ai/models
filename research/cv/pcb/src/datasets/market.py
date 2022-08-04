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
"""market.py"""

import os.path as osp
from glob import glob
import re

class Market:
    """Class for processing Market-1501 dataset"""
    def __init__(self, root, subset_name):
        """
        :param root: path of Market-1501 dataset
        :param subset_name: choose from ['train', 'query', 'gallery']
        """
        self.name2path = {'train': 'bounding_box_train',
                          'query': 'query',
                          'gallery': 'bounding_box_test'}
        self.images_dir = osp.join(root)
        self.subset_name = subset_name
        self.subset_path = self.name2path[subset_name]
        self.data = []
        self.num_ids = 0
        self.relabel = subset_name == 'train'
        self.load()

    def preprocess(self, path, relabel=True):
        """preprocess"""
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        fid = 0
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fpath, fid, pid, cam))
            fid += 1
        return ret, int(len(all_pids))

    def load(self):
        """load"""
        self.data, self.num_ids = self.preprocess(self.subset_path, self.relabel)
        print(self.__class__.__name__, "dataset loaded")
        print(" # subset | # ids | # images")
        print("  ---------------------------")
        print("  {:8} | {:5d} | {:8d}"
              .format(self.subset_name, self.num_ids, len(self.data)))
