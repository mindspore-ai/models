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
"""build training dataset"""
import os

import imageio
import numpy as np

from model_utils.config import config as cfg
from model_utils.data_file_utils import read_pickle, save_pickle, read_pose
from src.evaluation_dataset import Projector, PoseTransformer, LineModModelDB

linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone',
                     'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']


class LineModImageDB:
    """linemod image data parser"""
    def __init__(self, dataset_path, cls_name, render_num=10000, fuse_num=10000, ms_num=10000,
                 has_render_set=True, has_fuse_set=True):
        """__init__"""
        self.cls_name = cls_name
        self.dataset_path = dataset_path
        # some dirs for processing
        self.linemod_dir = os.path.join(self.dataset_path, 'LINEMOD')
        self.render_dir = 'renders/{}'.format(cls_name)
        self.rgb_dir = '{}/JPEGImages'.format(cls_name)
        self.mask_dir = '{}/mask'.format(cls_name)
        self.rt_dir = os.path.join(self.dataset_path, 'LINEMOD_ORIG', cls_name, 'data')
        self.render_num = render_num

        self.test_fn = '{}/test.txt'.format(cls_name)
        self.train_fn = '{}/train.txt'.format(cls_name)
        self.val_fn = '{}/val.txt'.format(cls_name)

        if has_render_set:
            self.render_pkl = os.path.join(self.linemod_dir, 'posedb', '{}_render.pkl'.format(cls_name))
            # prepare dataset
            if os.path.exists(self.render_pkl):
                # read cached
                self.render_set = read_pickle(self.render_pkl)
            else:
                # process render set
                self.render_set = self.collect_render_set_info(self.render_pkl, self.render_dir)
        else:
            self.render_set = []

        self.real_pkl = os.path.join(self.linemod_dir, 'posedb', '{}_real.pkl'.format(cls_name))
        if os.path.exists(self.real_pkl):
            # read cached
            self.real_set = read_pickle(self.real_pkl)
        else:
            # process real set
            self.real_set = self.collect_real_set_info()

        # prepare train test split
        self.train_real_set = []
        self.test_real_set = []
        self.val_real_set = []
        self.collect_train_val_test_info()

        self.fuse_set = []
        self.fuse_dir = 'fuse'
        self.fuse_num = fuse_num
        self.cls_idx = linemod_cls_names.index(cls_name)

        if has_fuse_set:
            self.fuse_pkl = os.path.join(self.linemod_dir, 'posedb', '{}_fuse.pkl'.format(cls_name))
            # prepare dataset
            if os.path.exists(self.fuse_pkl):
                # read cached
                self.fuse_set = read_pickle(self.fuse_pkl)
            else:
                # process render set
                self.fuse_set = self.collect_fuse_info()
        else:
            self.fuse_set = []

    def collect_render_set_info(self, pkl_file, render_dir, fmt='jpg'):
        """ rander dataset"""
        database = []
        projector = Projector()
        model_db = LineModModelDB()
        for k in range(self.render_num):
            print(k)
            data = {}
            data['rgb_pth'] = os.path.join(render_dir, '{}.{}'.format(k, fmt))
            data['dpt_pth'] = os.path.join(render_dir, '{}_depth.png'.format(k))
            data['RT'] = read_pickle(os.path.join(self.linemod_dir, render_dir, '{}_RT.pkl'.format(k)))['RT']
            data['cls_typ'] = self.cls_name
            data['rnd_typ'] = 'render'
            data['corners'] = projector.project(model_db.get_corners_3d(self.cls_name), data['RT'], 'blender')
            data['farthest'] = projector.project(model_db.get_farthest_3d(self.cls_name), data['RT'], 'blender')
            data['center'] = projector.project(model_db.get_centers_3d(self.cls_name)[None, :], data['RT'], 'blender')
            for num in [4, 12, 16, 20]:
                data['farthest{}'.format(num)] = projector.project(model_db.get_farthest_3d(self.cls_name, num),
                                                                   data['RT'], 'blender')
            data['small_bbox'] = projector.project(model_db.get_small_bbox(self.cls_name), data['RT'], 'blender')
            axis_direct = np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts'] = projector.project_h(axis_direct, data['RT'], 'blender')
            database.append(data)

        save_pickle(database, pkl_file)
        return database

    def collect_real_set_info(self):
        """ real image dataset"""
        database = []
        projector = Projector()
        model_db = LineModModelDB()
        img_num = len(os.listdir(os.path.join(self.linemod_dir, self.rgb_dir)))
        for k in range(img_num):
            data = {}
            data['rgb_pth'] = os.path.join(self.rgb_dir, '{:06}.jpg'.format(k))
            data['dpt_pth'] = os.path.join(self.mask_dir, '{:04}.png'.format(k))
            pose = read_pose(os.path.join(self.rt_dir, 'rot{}.rot'.format(k)),
                             os.path.join(self.rt_dir, 'tra{}.tra'.format(k)))
            pose_transformer = PoseTransformer(class_type=self.cls_name)
            data['RT'] = pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)
            data['cls_typ'] = self.cls_name
            data['rnd_typ'] = 'real'
            data['corners'] = projector.project(model_db.get_corners_3d(self.cls_name), data['RT'], 'linemod')
            data['farthest'] = projector.project(model_db.get_farthest_3d(self.cls_name), data['RT'], 'linemod')
            for num in [4, 12, 16, 20]:
                data['farthest{}'.format(num)] = projector.project(model_db.get_farthest_3d(self.cls_name, num),
                                                                   data['RT'], 'linemod')
            data['center'] = projector.project(model_db.get_centers_3d(self.cls_name)[None, :], data['RT'], 'linemod')
            data['small_bbox'] = projector.project(model_db.get_small_bbox(self.cls_name), data['RT'], 'linemod')
            axis_direct = np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts'] = projector.project_h(axis_direct, data['RT'], 'linemod')
            database.append(data)

        save_pickle(database, self.real_pkl)
        return database

    def collect_train_val_test_info(self):
        """ train and valid dataset"""
        with open(os.path.join(self.linemod_dir, self.test_fn), 'r') as f:
            test_fns = [line.strip().split('/')[-1] for line in f.readlines()]

        with open(os.path.join(self.linemod_dir, self.train_fn), 'r') as f:
            train_fns = [line.strip().split('/')[-1] for line in f.readlines()]

        with open(os.path.join(self.linemod_dir, self.val_fn), 'r') as f:
            val_fns = [line.strip().split('/')[-1] for line in f.readlines()]

        for data in self.real_set:
            if data['rgb_pth'].split('/')[-1] in test_fns:
                if data['rgb_pth'].split('/')[-1] in val_fns:
                    self.val_real_set.append(data)
                else:
                    self.test_real_set.append(data)

            if data['rgb_pth'].split('/')[-1] in train_fns:
                self.train_real_set.append(data)

    def collect_fuse_info(self):
        """ fuse dataset"""
        database = []
        model_db = LineModModelDB()
        projector = Projector()
        for k in range(self.fuse_num):
            data = dict()
            data['rgb_pth'] = os.path.join(self.fuse_dir, '{}_rgb.jpg'.format(k))
            data['dpt_pth'] = os.path.join(self.fuse_dir, '{}_mask.png'.format(k))

            # if too few foreground pts then continue
            mask = imageio.imread(os.path.join(self.linemod_dir, data['dpt_pth']))
            if np.sum(mask == (cfg.linemod_cls_names.index(self.cls_name) + 1)) < 400: continue

            data['cls_typ'] = self.cls_name
            data['rnd_typ'] = 'fuse'
            begins, poses = read_pickle(os.path.join(self.linemod_dir, self.fuse_dir, '{}_info.pkl'.format(k)))
            data['RT'] = poses[self.cls_idx]
            K = projector.intrinsic_matrix['linemod'].copy()
            K[0, 2] += begins[self.cls_idx, 1]
            K[1, 2] += begins[self.cls_idx, 0]
            data['K'] = K
            data['corners'] = projector.project_K(model_db.get_corners_3d(self.cls_name), data['RT'], K)
            data['center'] = projector.project_K(model_db.get_centers_3d(self.cls_name), data['RT'], K)
            data['farthest'] = projector.project_K(model_db.get_farthest_3d(self.cls_name), data['RT'], K)
            for num in [4, 12, 16, 20]:
                data['farthest{}'.format(num)] = projector.project_K(model_db.get_farthest_3d(self.cls_name, num),
                                                                     data['RT'], K)
            data['small_bbox'] = projector.project_K(model_db.get_small_bbox(self.cls_name), data['RT'], K)
            database.append(data)

        save_pickle(database, self.fuse_pkl)
        return database


def generateposedb(cls_name, pvnet_path):
    """generate pose database"""
    LineModImageDB(pvnet_path, cls_name, has_fuse_set=True, has_render_set=True)


if __name__ == "__main__":
    # classes to generate posedb file
    cls_list = ["cat"]
    for item in cls_list:
        cfg.eval_dataset = '/data/bucket-4609/dataset/pvnet/data'
        generateposedb(item, cfg.eval_dataset)
