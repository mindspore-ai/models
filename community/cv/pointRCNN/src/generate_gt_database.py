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
# This file was copied from project [sshaoshuai][https://github.com/sshaoshuai/PointRCNN]
"""generate groundtruth database"""
import os
import argparse
import pickle
import numpy as np
import torch

import src.lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from src.lib.datasets.kitti_dataset import KittiDataset


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./gt_database')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


class GTDatabaseGenerator(KittiDataset):
    """GTDatabaseGenerator"""
    def __init__(self, root_dir, split='train', classes=args.class_name):
        super().__init__(root_dir, split=split)
        self.gt_database = None
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        """filtrate_objects"""
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self):
        """generate groundtruth database"""
        gt_database = []
        for _, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            print('process gt sample (id=%06d)' % sample_id)

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]

            obj_list = self.filtrate_objects(self.get_label(sample_id))

            gt_boxes3d = np.zeros((len(obj_list), 7), dtype=np.float32)
            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry

            if not gt_boxes3d:
                print('No gt object')
                continue

            boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(
                torch.from_numpy(pts_rect), torch.from_numpy(gt_boxes3d))

            for k in range(len(boxes_pts_mask_list)):
                pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_intensity = pts_intensity[pt_mask_flag].astype(
                    np.float32)
                sample_dict = {
                    'sample_id': sample_id,
                    'cls_type': obj_list[k].cls_type,
                    'gt_box3d': gt_boxes3d[k],
                    'points': cur_pts,
                    'intensity': cur_pts_intensity,
                    'obj': obj_list[k]
                }
                gt_database.append(sample_dict)

        save_file_name = os.path.join(
            args.save_dir,
            '%s_gt_database_3level_%s.pkl' % (args.split, self.classes[-1]))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)


if __name__ == '__main__':
    dataset = GTDatabaseGenerator(root_dir='../data/', split=args.split)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.generate_gt_database()
