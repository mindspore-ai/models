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
"""single dataset"""

import os.path
from src.data.image_folder import make_dataset
from src.data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np


class SingleDataset(BaseDataset):
    """SingleDataset"""
    def __init__(self, opt):
        super(SingleDataset, self).__init__()
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot)
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)  # A: numpy
        A = np.squeeze(A, axis=0)
        return A, np.array(A_path, np.str)

    def __len__(self):
        return len(self.A_paths)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'SingleImageDataset'
