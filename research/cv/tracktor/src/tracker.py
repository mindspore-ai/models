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
"""Tracker"""

from collections import deque

import cv2
import numpy as np

from src.tracking_utils import clip_boxes_to_image
from src.tracking_utils import get_center
from src.tracking_utils import get_height
from src.tracking_utils import get_width
from src.tracking_utils import make_pos
from src.tracking_utils import nms
from src.tracking_utils import warp_pos


class Tracker:
    """
    Tracker class, handle all tracking logic.

    Args:
        obj_detect (FasterRCNNTrackerWrapped): Feature Extractor of Faster RCNN
        reid_network (nn.Cell): Network for evaluate appearance features
        tracker_cfg: Tracker configuration dictionary.
    """

    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg.detection_person_thresh
        self.regression_person_thresh = tracker_cfg.regression_person_thresh
        self.detection_nms_thresh = tracker_cfg.detection_nms_thresh
        self.regression_nms_thresh = tracker_cfg.regression_nms_thresh
        self.inactive_patience = tracker_cfg.inactive_patience
        self.max_features_num = tracker_cfg.max_features_num
        self.reid_sim_threshold = tracker_cfg.reid_sim_threshold
        self.reid_iou_threshold = tracker_cfg.reid_iou_threshold
        self.do_align = tracker_cfg.do_align
        self.motion_model_cfg = tracker_cfg.motion_model

        self.warp_mode = getattr(cv2, tracker_cfg.warp_mode)
        self.number_of_iterations = tracker_cfg.number_of_iterations
        self.termination_eps = tracker_cfg.termination_eps

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

    def reset(self, hard=True):
        """Reset tracker state"""
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        """Add tracks to inactive category."""
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.shape[0]
        for i in range(num_new):
            self.tracks.append(Track(
                np.reshape(new_det_pos[i], (1, -1)),
                new_det_scores[i],
                self.track_num + i,
                np.reshape(new_det_features[i], (1, -1)),
                self.inactive_patience,
                self.max_features_num,
                self.motion_model_cfg.n_steps if self.motion_model_cfg.n_steps > 0 else 1
            ))
        self.track_num += num_new

    def regress_tracks(self, blob, boxes, scores):
        """Regress the position of the tracks and also checks their scores."""
        pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = np.reshape(pos[i], (1, -1))

        return np.asarray(s[::-1])

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = np.concatenate([t.pos for t in self.tracks], 0)
        else:
            pos = np.zeros(0)
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = np.concatenate([t.features for t in self.tracks], 0)
        else:
            features = np.zeros(0)
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = np.concatenate([t.features for t in self.inactive_tracks], 0)
        else:
            features = np.zeros(0)
        return features

    def get_appearances(self, blob, pos):
        """Uses the siamese CNN to get the features for all active tracks."""
        raise NotImplementedError('get appearance not implemented')

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(np.reshape(f, (1, -1)))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image, (1, 2, 0))
            im2 = np.transpose(blob['img'][0], (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations, self.termination_eps)
            _, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            warp_matrix = warp_matrix

            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg.enabled:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg.center_only:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg.center_only:
                vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = np.stack(vs).mean(axis=0)
            self.motion_step(t)

    def process_new_detections(self, image, boxes, scores):
        """Process boxes from list of new detection"""
        if boxes.size > 0:
            boxes = clip_boxes_to_image(boxes, image.shape[-2:])
            inds = np.reshape(np.stack(np.greater(scores, self.detection_person_thresh).nonzero(), axis=-1), (-1,))
        else:
            inds = np.zeros(0)

        if inds.size > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = np.zeros(0)
            det_scores = np.zeros(0)

        return det_pos, det_scores

    def step(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.copy())

        if self.tracks:
            # align
            if self.do_align:
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg.enabled:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

        dets = blob['dets'].squeeze(axis=0)
        if self.tracks:
            pos = self.get_pos()
        else:
            pos = np.zeros(0)

        boxes, scores, r_boxes, r_scores = self.obj_detect.process_images_and_boxes(
            blob['img'], dets, pos,
        )

        det_pos, det_scores = self.process_new_detections(blob['img'], boxes, scores)

        if self.tracks:
            # create nms input
            person_scores = self.regress_tracks(blob, r_boxes, r_scores)
            # nms here if tracks overlap
            keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

            self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

        # Here NMS is used to filter out detections that are already covered by tracks. This is
        # done by iterating through the active tracks one by one, assigning them a bigger score
        # than 1 (maximum score for detections) and then filtering the detections with NMS.
        # In the paper this is done by calculating the overlap with existing tracks, but the
        # result stays the same.
        if det_pos.size > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            det_pos, det_scores = self.check_intersection_with_tracks(det_pos, det_scores)

        if det_pos.size > 0:
            self.add_new_tracks(det_pos, det_scores)

        self.generate_results(blob)

    def check_intersection_with_tracks(self, det_pos, det_scores):
        """check with every track in a single run (problem if tracks delete each other)"""
        for t in self.tracks:
            nms_track_pos = np.concatenate([t.pos, det_pos])
            nms_track_scores = np.concatenate(
                [np.asarray([2.0]), det_scores]
            )
            keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

            keep = keep[np.greater_equal(keep, 1)] - 1

            det_pos = det_pos[keep]
            det_scores = det_scores[keep]
            if keep.size == 0:
                break
        return det_pos, det_scores

    def add_new_tracks(self, det_pos, det_scores):
        """create new tracks"""
        new_det_pos = det_pos
        new_det_scores = det_scores
        new_det_features = [np.zeros(0) for _ in range(len(new_det_pos))]

        if new_det_pos.size > 0:
            self.add(new_det_pos, new_det_scores, new_det_features)

    def generate_results(self, blob):
        """Generate resulting tracking prediction"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate(
                [t.pos[0], np.array([t.score])]
            )

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]

    def get_results(self):
        """Get tracking results"""
        return self.results


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        np.ndarray: distance matrix.
    """
    m, n = input1.shape[0], input2.shape[0]
    mat1 = np.broadcast_to((input1 ** 2).sum(axis=1, keepdims=True), (m, n))
    mat2 = np.transpose(np.broadcast_to((input2 ** 2).sum(axis=1, keep_dims=True), (n, m)), (1, 0))
    distmat = mat1 + mat2
    distmat = np.matmul(input1, np.transpose(input2, (1, 0))) + distmat
    return distmat


class Track:
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([np.copy(pos)], maxlen=mm_steps + 1)
        self.last_v = np.asarray([])
        self.gt_id = None

    def has_positive_area(self):
        """Check if track bbox has positive area."""
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object."""
        if len(self.features) > 1:
            features = np.concatenate(list(self.features), axis=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdims=True)
        dist = euclidean_squared_distance(features, test_features)
        return dist

    def reset_last_pos(self):
        """Reset last position of track."""
        self.last_pos.clear()
        self.last_pos.append(self.pos.copy())
