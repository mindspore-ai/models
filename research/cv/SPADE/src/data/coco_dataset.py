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
""" Dataset: COCO """

import os.path
from src.data.pix2pix_dataset import Pix2pixDataset
from src.data.image_folder import make_dataset


class CocoDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser

    def init_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else opt.phase

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            label_portrait_dir = os.path.join(root, '%s_label_portrait' % phase)
            if os.path.isdir(label_portrait_dir):
                label_portrait_paths = make_dataset(label_portrait_dir, recursive=False, read_cache=True)
                label_paths += label_portrait_paths

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            image_portrait_dir = os.path.join(root, '%s_img_portrait' % phase)
            if os.path.isdir(image_portrait_dir):
                image_portrait_paths = make_dataset(image_portrait_dir, recursive=False, read_cache=True)
                image_paths += image_portrait_paths

        if not opt.no_instance:
            instance_dir = os.path.join(root, '%s_inst' % phase)
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

            if not opt.coco_no_portraits and opt.isTrain:
                instance_portrait_dir = os.path.join(root, '%s_inst_portrait' % phase)
                if os.path.isdir(instance_portrait_dir):
                    instance_portrait_paths = make_dataset(instance_portrait_dir, recursive=False, read_cache=True)
                    instance_paths += instance_portrait_paths

        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths
