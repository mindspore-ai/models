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

# This file was copied from project
# [Charles R. Qi][https://github.com/simon3dv/frustum_pointnets_pytorch]
''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import kitti_util as utils

raw_input = input
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))


class kitti_object():
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == 'training'
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass


def get_lidar_in_image_fov(pc_velo,
                           calib,
                           xmin,
                           ymin,
                           xmax,
                           ymax,
                           return_more=False,
                           clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & (
        pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    return imgfov_pc_velo
    