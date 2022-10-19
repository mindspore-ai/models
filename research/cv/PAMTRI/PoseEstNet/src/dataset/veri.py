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
"""veri"""
import os
import csv
import math
from collections import OrderedDict
import numpy as np
from .JointsDataset import JointsDataset

class VeRiDataset(JointsDataset):
    """VeRiDataset"""
    def __init__(self, root, is_train, transform=None):
        super(VeRiDataset, self).__init__(root, is_train, transform)
        self.num_joints = 36
        self.flip_pairs = [[0, 18], [1, 19], [2, 20], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25], [8, 26], [9, 27],
                           [10, 28], [11, 29], [12, 30], [13, 31], [14, 32], [15, 33], [16, 34], [17, 35]]
        self.image_width = 256
        self.image_height = 256
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.db = self._get_db()

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', 'label_' + self.image_set + '.csv'
        )
        print(file_name)

        hash_annot = {}
        with open(file_name) as annot_file:
            reader = csv.reader(annot_file, delimiter=',')
            for row in reader:
                img_name = row[0]
                width = int(row[1])
                height = int(row[2])
                joints = []
                for j in range(36):
                    joint = [int(row[j*3+3]), int(row[j*3+4]), int(row[j*3+5])]
                    joints.append(joint)
                hash_annot[img_name] = (width, height, joints)

        gt_db = []
        for k in sorted(hash_annot.keys()):
            image_name = k
            width = hash_annot[k][0]
            height = hash_annot[k][1]
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)

            joints = np.array(hash_annot[k][2])
            assert len(joints) == self.num_joints, \
                'joint num diff: {} vs {}'.format(len(joints),
                                                  self.num_joints)

            joints_3d[:, 0:2] = joints[:, 0:2]
            joints_3d_vis[:, 0] = joints[:, 2]
            joints_3d_vis[:, 1] = joints[:, 2]

            center = np.zeros((2), dtype=np.float32)
            center[0] = width * 0.5
            center[1] = height* 0.5

            if width > self.aspect_ratio * height:
                height = width * 1.0 / self.aspect_ratio
            elif width < self.aspect_ratio * height:
                width = height * self.aspect_ratio
            scale = np.array(
                [width * 1.0 / self.pixel_std, height * 1.0 / self.pixel_std],
                dtype=np.float32)
            if center[0] != -1:
                scale = scale * 1.25

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'

            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, 'image_' + self.image_set, image_name),
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, preds, output_dir, *args, **kwargs):
        SC_BIAS = 0.25
        threshold = 0.5

        gt_file = os.path.join(self.root,
                               'annot',
                               'label_{}.csv'.format('test'))

        gts = []
        viss = []
        area_sqrts = []
        with open(gt_file) as annot_file:
            reader = csv.reader(annot_file, delimiter=',')
            for row in reader:
                joints = []
                vis = []
                top_lft = btm_rgt = [int(float(row[3])), int(float(row[4]))]
                for j in range(36):
                    joint = [int(float(row[j*3+3])), int(float(row[j*3+4])), int(float(row[j*3+5]))]
                    joints.append(joint)
                    vis.append(joint[2])
                    if joint[0] < top_lft[0]:
                        top_lft[0] = joint[0]
                    if joint[1] < top_lft[1]:
                        top_lft[1] = joint[1]
                    if joint[0] > btm_rgt[0]:
                        btm_rgt[0] = joint[0]
                    if joint[1] > btm_rgt[1]:
                        btm_rgt[1] = joint[1]
                gts.append(joints)
                viss.append(vis)
                area_sqrts.append(math.sqrt((btm_rgt[0] - top_lft[0] + 1) * (btm_rgt[1] - top_lft[1] + 1)))

        jnt_visible = np.array(viss, dtype=np.int)
        jnt_visible = np.transpose(jnt_visible)
        pos_pred_src = np.transpose(preds, [1, 2, 0])
        pos_gt_src = np.transpose(gts, [1, 2, 0])
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        area_sqrts = np.linalg.norm(area_sqrts, axis=0)
        area_sqrts *= SC_BIAS
        scale = np.multiply(area_sqrts, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 36))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Wheel', (1.0/4.0) * (PCKh[0] + PCKh[1] + PCKh[18] + PCKh[19])),
            ('Fender', (1.0/16.0) * (PCKh[2] + PCKh[3] + PCKh[4] + PCKh[5] + PCKh[6] + PCKh[7] + PCKh[8] +
                                     PCKh[9] + PCKh[20] + PCKh[21] + PCKh[22] + PCKh[23] + PCKh[24] +
                                     PCKh[25] + PCKh[26] + PCKh[27])),
            ('Back', (1.0/4.0) * (PCKh[10] + PCKh[11] + PCKh[28] + PCKh[29])),
            ('Front', (1.0/4.0) * (PCKh[16] + PCKh[17] + PCKh[34] + PCKh[35])),
            ('WindshieldBack', (1.0/4.0) * (PCKh[12] + PCKh[13] + PCKh[30] + PCKh[31])),
            ('WindshieldFront', (1.0/4.0) * (PCKh[14] + PCKh[15] + PCKh[32] + PCKh[33])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
