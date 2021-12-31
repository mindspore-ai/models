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
"""eval dataset"""
import os

import numpy as np
from plyfile import PlyData

from model_utils.config import config as cfg


class Projector:
    """Projector"""
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]]),
        'blender': np.array([[700., 0., 320.],
                             [0., 700., 240.],
                             [0., 0., 1.]]),
        'pascal': np.asarray([[-3000.0, 0.0, 0.0],
                              [0.0, 3000.0, 0.0],
                              [0.0, 0.0, 1.0]])
    }

    def project(self, pts_3d, rt_matrix, k_type):
        """project"""
        pts_2d = np.matmul(pts_3d, rt_matrix[:, :3].T) + rt_matrix[:, 3:].T
        pts_2d = np.matmul(pts_2d, self.intrinsic_matrix[k_type].T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d

    def project_h(self, pts_3dh, rt_matrix, k_type):
        """
        :param pts_3dh: [n,4]
        :param rt_matrix:      [3,4]
        :param k_type:
        :return: [n,3]
        """
        K = self.intrinsic_matrix[k_type]
        return np.matmul(np.matmul(pts_3dh, rt_matrix.transpose()), K.transpose())

    def project_pascal(self, pts_3d, rt_matrix, principle):
        """
        :param pts_3d:    [n,3]
        :param rt_matrix:        [3,4]
        :param principle: [2,2]
        :return:
        """
        K = self.intrinsic_matrix['pascal'].copy()
        K[:2, 2] = principle
        cam_3d = np.matmul(pts_3d, rt_matrix[:, :3].T) + rt_matrix[:, 3:].T
        cam_3d[np.abs(cam_3d[:, 2]) < 1e-5, 2] = 1e-5  # revise depth
        pts_2d = np.matmul(cam_3d, K.T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d, cam_3d

    def project_pascal_h(self, pts_3dh, rt_matrix, principle):
        """project_pascal_h"""
        K = self.intrinsic_matrix['pascal'].copy()
        K[:2, 2] = principle
        return np.matmul(np.matmul(pts_3dh, rt_matrix.transpose()), K.transpose())

    @staticmethod
    def project_K(pts_3d, rt_matrix, k_matrix):
        """project_K"""
        pts_2d = np.matmul(pts_3d, rt_matrix[:, :3].T) + rt_matrix[:, 3:].T
        pts_2d = np.matmul(pts_2d, k_matrix.T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d


class ModelAligner:
    """ModelAligner"""
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {
        # 'cat': np.array([-0.00577495, -0.01259045, -0.04062323])
    }
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]]),
        'blender': np.array([[700., 0., 320.],
                             [0., 700., 240.],
                             [0., 0., 1.]])
    }

    def __init__(self, class_type='cat'):
        """__init__"""
        self.class_type = class_type
        self.blender_model_path = os.path.join(cfg.dataset_dir, cfg.dataset_name, '{0}/{0}.ply'.format(class_type))
        self.orig_model_path = os.path.join(cfg.dataset_dir, cfg.origin_dataset_name, '{}/mesh.ply'.format(class_type))
        self.orig_old_model_path = os.path.join(cfg.dataset_dir, cfg.origin_dataset_name,
                                                '{}/OLDmesh.ply'.format(class_type))
        self.transform_dat_path = os.path.join(cfg.dataset_dir, cfg.origin_dataset_name,
                                               '{}/transform.dat'.format(class_type))

        self.R_p2w, self.t_p2w, self.s_p2w = self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        """setup_p2w_transform"""
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                               [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                               [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                               [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                               [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        # printer system to world system
        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w, t_p2w, s_p2w

    def pose_p2w(self, RT):
        """pose_p2w"""
        t, R = RT[:, 3], RT[:, :3]
        R_w2c = np.dot(R, self.R_p2w.T)
        t_w2c = -np.dot(R_w2c, self.t_p2w) + self.s_p2w * t
        return np.concatenate([R_w2c, t_w2c[:, None]], 1)

    @staticmethod
    def load_ply_model(model_path):
        """load_ply_model"""
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def read_transform_dat(self):
        """read_transform_dat"""
        transform_dat = np.loadtxt(self.transform_dat_path, skiprows=1)[:, 1]
        transform_dat = np.reshape(transform_dat, newshape=[3, 4])
        return transform_dat

    def load_orig_model(self):
        """load_orig_model"""
        if os.path.exists(self.orig_model_path):
            return self.load_ply_model(self.orig_model_path) / 1000.
        transform = self.read_transform_dat()
        old_model = self.load_ply_model(self.orig_old_model_path) / 1000.
        old_model = np.dot(old_model, transform[:, :3].T) + transform[:, 3]
        return old_model

    def get_translation_transform(self):
        """get_translation_transform"""
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path)
        orig_model = self.load_orig_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform


class PoseTransformer:
    """PoseTransformer"""
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models = {}

    def __init__(self, class_type):
        """__init__"""
        self.class_type = class_type
        self.blender_model_path = os.path.join(cfg.dataset_dir, cfg.dataset_name, '{0}/{0}.ply'.format(class_type))
        self.orig_model_path = os.path.join(cfg.dataset_dir, cfg.origin_dataset_name, '{}/mesh.ply'.format(class_type))
        self.model_aligner = ModelAligner(class_type)

    def orig_pose_to_blender_pose(self, pose):
        """orig_pose_to_blender_pose"""
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


class LineModModelDB:
    """
    LineModModelDB is used for managing the mesh of each model
    """
    corners_3d = {}
    models = {}
    diameters = {}
    centers_3d = {}
    farthest_3d = {'8': {}, '4': {}, '12': {}, '16': {}, '20': {}}
    small_bbox_corners = {}

    def __init__(self):
        """__init__"""
        self.ply_pattern = os.path.join(cfg.eval_dataset, cfg.dataset_name, '{}/{}.ply')
        self.diameter_pattern = os.path.join(cfg.eval_dataset, cfg.origin_dataset_name, '{}/distance.txt')
        self.farthest_pattern = os.path.join(cfg.eval_dataset, cfg.dataset_name, '{}/farthest{}.txt')

    def get_corners_3d(self, class_type):
        """get_corners_3d"""
        if class_type in self.corners_3d:
            return self.corners_3d[class_type]

        corner_pth = os.path.join(cfg.eval_dataset, cfg.dataset_name, class_type, 'corners.txt')
        if os.path.exists(corner_pth):
            self.corners_3d[class_type] = np.loadtxt(corner_pth)
            return self.corners_3d[class_type]

        ply_path = self.ply_pattern.format(class_type, class_type)
        ply = PlyData.read(ply_path)
        data = ply.elements[0].data

        x = data['x']
        min_x, max_x = np.min(x), np.max(x)
        y = data['y']
        min_y, max_y = np.min(y), np.max(y)
        z = data['z']
        min_z, max_z = np.min(z), np.max(z)
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        self.corners_3d[class_type] = corners_3d
        np.savetxt(corner_pth, corners_3d)

        return corners_3d

    def get_small_bbox(self, class_type):
        """get_small_bbox"""
        if class_type in self.small_bbox_corners:
            return self.small_bbox_corners[class_type]

        corners = self.get_corners_3d(class_type)
        center = np.mean(corners, 0)
        small_bbox_corners = (corners - center[None, :]) * 2.0 / 3.0 + center[None, :]
        self.small_bbox_corners[class_type] = small_bbox_corners

        return small_bbox_corners

    def get_ply_model(self, class_type):
        """get_ply_model"""
        if class_type in self.models:
            return self.models[class_type]

        ply = PlyData.read(self.ply_pattern.format(class_type, class_type))
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        self.models[class_type] = model
        return model

    def get_diameter(self, class_type):
        """get_diameter"""
        if class_type in self.diameters:
            return self.diameters[class_type]

        diameter_path = self.diameter_pattern.format(class_type)
        diameter = np.loadtxt(diameter_path) / 100.
        self.diameters[class_type] = diameter
        return diameter

    def get_centers_3d(self, class_type):
        """get_centers_3d"""
        if class_type in self.centers_3d:
            return self.centers_3d[class_type]

        c3d = self.get_corners_3d(class_type)
        self.centers_3d[class_type] = (np.max(c3d, 0) + np.min(c3d, 0)) / 2
        return self.centers_3d[class_type]

    def get_farthest_3d(self, class_type, num=8):
        """get_farthest_3d"""
        if class_type in self.farthest_3d['{}'.format(num)]:
            return self.farthest_3d['{}'.format(num)][class_type]

        if num == 8:
            farthest_path = self.farthest_pattern.format(class_type, '')
        else:
            farthest_path = self.farthest_pattern.format(class_type, num)
        farthest_pts = np.loadtxt(farthest_path)
        self.farthest_3d['{}'.format(num)][class_type] = farthest_pts
        return farthest_pts

    def get_ply_mesh(self, class_type):
        """get_ply_mesh"""
        ply = PlyData.read(self.ply_pattern.format(class_type, class_type))
        vert = np.asarray([ply['vertex'].data['x'], ply['vertex'].data['y'], ply['vertex'].data['z']]).transpose()
        vert_id = [vid for vid in ply['face'].data['vertex_indices']]
        vert_id = np.asarray(vert_id, np.int64)

        return vert, vert_id


def get_pts_3d(class_type):
    """get_pts_3d"""
    linemod_db = LineModModelDB()
    points_3d = linemod_db.get_farthest_3d(class_type)
    points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
    return points_3d
