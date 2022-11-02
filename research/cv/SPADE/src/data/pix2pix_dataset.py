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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" spade dataset itor """

import os
import numpy as np
from PIL import Image
from src.data.base_dataset import BaseDataset, get_params, get_transform
from src.util import util

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        label_paths, image_paths, instance_paths = self.init_paths(opt)
        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def init_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.init_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def get_paths(self, opt):
        return self.label_paths, self.image_paths, self.instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        label_np = np.array(get_transform(self.opt, label, params, method=Image.NEAREST))
        if label_np.ndim == 2:
            label_np = np.expand_dims(label_np, axis=0)
        else:
            label_np = np.rollaxis(label_np, 2, 0)
        label_np[label_np == 255] = self.opt.label_nc

        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        image_np = np.array(get_transform(self.opt, image, params), np.float32)
        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, axis=0) / 255.0
        else:
            image_np = np.rollaxis(image_np, 2, 0) / 255.0
        image_np = (image_np - 0.5) / 0.5

        # if using instance maps
        if self.opt.no_instance:
            instance_np = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            instance_np = get_transform(self.opt, instance, params, method=Image.NEAREST)
            if instance_np.ndim == 2:
                instance_np = np.expand_dims(instance_np, axis=0)
            else:
                instance_np = np.rollaxis(instance_np, 2, 0)
            if instance.mode == 'L':
                instance_np = instance_np.astype("int64")
            else:
                instance_np = instance_np / 255.0

        # Give subclasses a chance to modify the final output
        self.postprocess(label_np)
        return label_np, instance_np, image_np

    def postprocess(self, label):
        return label

    def geta(self, index):
        return self.__getitem__(index)

    def __len__(self):
        return self.dataset_size
