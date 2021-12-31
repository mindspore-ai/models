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
"""
Useful functions
"""
import random
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

random.seed(10)

########################################[ Scheduler ]########################################


def poly_lr(epoch, epoch_max=30, lr=1e-4, power=0.9, cutoff_epoch=29):
    return lr * (1 - (1.0 * min(epoch, cutoff_epoch) / epoch_max)) ** power


class PolyLR:
    """polynomial learning rate scheduler"""

    def __init__(self, epoch_max=30, base_lr=1e-4, power=0.9, cutoff_epoch=29):
        self.epoch, self.epoch_max, self.base_lr, self.power, self.cutoff_epoch = (
            0,
            epoch_max,
            base_lr,
            power,
            cutoff_epoch,
        )

    def get_lr(self):
        return (
            self.base_lr
            * (1 - (1.0 * min(self.epoch, self.cutoff_epoch) / self.epoch_max))
            ** self.power
        )

    def step(self):
        self.epoch = self.epoch + 1


########################################[ General ]########################################


def get_points_mask(size, points):
    """ generate point mask from point list """
    mask = np.zeros(size[::-1]).astype(np.uint8)
    if list(points):
        points = np.array(points)
        mask[points[:, 1], points[:, 0]] = 1
    return mask


def get_points_list(mask):
    """ generate point list from point mask """
    pts_y, pts_x = np.where(mask == 1)
    pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
    return pts_xy.tolist()


########################################[ Robot Strategy ]########################################


def get_anno_point(pred, gt, anno_points):
    """ get next click for robot user"""
    fn_map, fp_map = (
        (gt == 1) & (pred == 0),
        (gt == 0) & (pred == 1),
    )

    fn_map = np.pad(fn_map, ((1, 1), (1, 1)), "constant")
    fndist_map = distance_transform_edt(fn_map)
    fndist_map = fndist_map[1:-1, 1:-1]

    fp_map = np.pad(fp_map, ((1, 1), (1, 1)), "constant")
    fpdist_map = distance_transform_edt(fp_map)
    fpdist_map = fpdist_map[1:-1, 1:-1]

    if isinstance(anno_points, list):
        for pt in anno_points:
            fndist_map[pt[1], pt[0]] = fpdist_map[pt[1], pt[0]] = 0
    else:
        fndist_map[anno_points == 1] = 0
        fpdist_map[anno_points == 1] = 0

    if np.max(fndist_map) > np.max(fpdist_map):
        usr_map, if_pos = fndist_map, True
    else:
        usr_map, if_pos = fpdist_map, False

    [y_mlist, x_mlist] = np.where(usr_map == np.max(usr_map))
    pt_new = (x_mlist[0], y_mlist[0])
    return pt_new, if_pos


########################################[ Train Sample Strategy ]########################################


def get_pos_points_walk(gt, pos_point_num, step=0.2, margin=0.2):
    """ sample random positive clicks"""
    if pos_point_num == 0:
        return []

    pos_points = []
    choice_map_margin = (gt == 1).astype(np.int64)
    choice_map_margin = np.pad(choice_map_margin, ((1, 1), (1, 1)), "constant")
    dist_map_margin = distance_transform_edt(choice_map_margin)[1:-1, 1:-1]

    if isinstance(margin, list):
        margin = random.choice(margin)

    if 0 < margin < 1.0:
        margin = int(dist_map_margin.max() * margin)

    choice_map_margin = dist_map_margin > margin

    choice_map_step = np.ones_like(gt).astype(np.int64)
    choice_map_step = np.pad(choice_map_step, ((1, 1), (1, 1)), "constant")

    if isinstance(step, list):
        step = random.choice(step)

    if 0 < step < 1.0:
        step = int(np.sqrt((gt == 1).sum() / np.pi) * 2 * step)

    for _ in range(pos_point_num):
        dist_map_step = distance_transform_edt(choice_map_step)[1:-1, 1:-1]
        pts_y, pts_x = np.where((choice_map_margin) & (dist_map_step > step))
        pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
        if not list(pts_xy):
            break
        pt_new = tuple(pts_xy[random.randint(0, len(pts_xy) - 1), :])
        pos_points.append(pt_new)
        choice_map_step[pt_new[1] + 1, pt_new[0] + 1] = 0

    return pos_points


def get_neg_points_walk(gt, neg_point_num, margin_min=0.06, margin_max=0.48, step=0.2):
    """ sample random negative clicks"""
    if neg_point_num == 0:
        return []

    neg_points = []

    if isinstance(margin_min, list):
        margin_min = random.choice(margin_min)
    if isinstance(margin_max, list):
        margin_max = random.choice(margin_max)

    if (0 < margin_min < 1.0) and (0 < margin_max < 1.0):
        fg = (gt == 1).astype(np.int64)
        fg = np.pad(fg, ((1, 1), (1, 1)), "constant")
        dist_fg = distance_transform_edt(fg)[1:-1, 1:-1]
        margin_max = min(max(int(dist_fg.max() * margin_min), 3), 10) * (
            margin_max / margin_min
        )
        margin_min = min(max(int(dist_fg.max() * margin_min), 3), 10)

    choice_map_margin = (gt != 1).astype(np.int64)
    dist_map_margin = distance_transform_edt(choice_map_margin)
    choice_map_margin = (dist_map_margin > margin_min) & (dist_map_margin < margin_max)

    choice_map_step = np.ones_like(gt).astype(np.int64)
    choice_map_step = np.pad(choice_map_step, ((1, 1), (1, 1)), "constant")

    if isinstance(step, list):
        step = random.choice(step)

    if 0 < step < 1.0:
        step = int(np.sqrt((gt == 1).sum() / np.pi) * 2 * step)

    for _ in range(neg_point_num):
        dist_map_step = distance_transform_edt(choice_map_step)[1:-1, 1:-1]
        pts_y, pts_x = np.where((choice_map_margin) & (dist_map_step > step))
        pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
        if not list(pts_xy):
            break
        pt_new = tuple(pts_xy[random.randint(0, len(pts_xy) - 1), :])
        neg_points.append(pt_new)
        choice_map_step[pt_new[1] + 1, pt_new[0] + 1] = 0

    return neg_points
