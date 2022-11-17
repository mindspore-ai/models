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
"""eval utils"""
import numpy as np
import cv2

from src.evaluation_dataset import get_pts_3d, LineModModelDB, Projector
from model_utils.config import config as cfg


def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    """pnp"""
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)

    R, _ = cv2.Rodrigues(R_exp)
    return np.concatenate([R, t], axis=-1)


def find_nearest_point_idx(point, contour):
    """
    given a point, find the nearest index of point in a contour
    @point(a value, not index) is a vertex
    @contours
    return the index of a nearest point
    """
    idx = -1
    distance = float("inf")
    for i in range(0, len(contour) - 1):
        d = np.linalg.norm(point - contour[i])
        if d < distance:
            distance = d
            idx = i
    return idx


class Evaluator:
    """PvNet Evaluator"""
    def __init__(self):
        """__init__"""
        self.linemod_db = LineModModelDB()
        self.projector = Projector()
        self.projection_2d_recorder = []
        self.add_recorder = []
        self.cm_degree_5_recorder = []
        self.proj_mean_diffs = []
        self.add_dists = []
        self.uncertainty_pnp_cost = []

    def projection_2d(self, pose_pred, pose_targets, model, K, threshold=5):
        """projection_2d"""
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add_recorder.append(mean_dist < diameter)
        self.add_dists.append(mean_dist)

    def add_metric_sym(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """
        ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist = np.mean(find_nearest_point_idx(model_pred, model_targets))
        self.add_recorder.append(mean_dist < diameter)
        self.add_dists.append(mean_dist)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        """ 5 cm 5 degree metric
        pose_pred is considered correct if the translation and rotation errors are below 5 cm and 5 degree respectively
        """
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cm_degree_5_recorder.append(translation_distance < 5 and angular_distance < 5)

    def evaluate(self, points_2d, pose_targets, class_type):
        """evaluate"""
        points_3d = get_pts_3d(class_type)

        k_matrix = np.array([[572.41140, 0., 325.26110],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]], np.float32)

        pose_pred = pnp(points_3d, points_2d, k_matrix)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if cfg.cls_name in ['eggbox', 'glue']:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)
        self.projection_2d(pose_pred, pose_targets, model, k_matrix)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def average_precision(self, verbose=True):
        """average_precision"""
        if verbose:
            print('2d projections metric: {}'.format(np.mean(self.projection_2d_recorder)))
            print('ADD metric: {}'.format(np.mean(self.add_recorder)))
            print('5 cm 5 degree metric: {}'.format(np.mean(self.cm_degree_5_recorder)))

        return np.mean(self.projection_2d_recorder), np.mean(self.add_recorder), np.mean(self.cm_degree_5_recorder)
