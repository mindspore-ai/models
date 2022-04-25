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
"""Eval Dataset"""

import os
import numpy as np
from PIL import Image
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset as de
from mindspore.common import set_seed
set_seed(0)

class LFWDataset:
    def __init__(self, data_dir, pairs_path):

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(data_dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return np.array(pairs, dtype=object)

    def get_lfw_paths(self, lfw_dir):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    # Modified here
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            new_path = path + '.jpg'
        elif os.path.exists(path + '.png'):
            new_path = path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)
        return new_path

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """


        # Modified to open as PIL image in the first place
        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = Image.open(path_1), Image.open(path_2)
        return (img1, img2, issame)

    def __len__(self):
        return len(self.validation_images)

def get_lfw_dataloader(eval_root_dir, eval_pairs_path, eval_batch_size):

    data_transforms = [C.RandomResize(size=(224, 224)),
                       P.ToPIL(),
                       P.ToTensor(),
                       P.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]


    face_dataset = LFWDataset(data_dir=eval_root_dir, pairs_path=eval_pairs_path)
    print(face_dataset)

    dataloaders = de.GeneratorDataset(face_dataset, ['img1', 'img2', 'issame'], shuffle=True)

    dataloaders = dataloaders.map(input_columns=["img1"], operations=data_transforms)
    dataloaders = dataloaders.map(input_columns=["img2"], operations=data_transforms)

    dataloaders = dataloaders.batch(eval_batch_size, True)

    return dataloaders
