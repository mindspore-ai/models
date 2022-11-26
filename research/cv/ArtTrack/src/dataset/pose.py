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

import json
import os
import random as rand
from enum import Enum

import numpy as np
from numpy import array as arr, concatenate as cat
import scipy.io as sio
from imageio import imread
from PIL import Image

from src import dataset
from src.log import log as logging


class Batch(Enum):
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    pairwise_targets = 5
    pairwise_mask = 6
    data_item = 7


def mirror_joints_map(all_joints, num_joints):
    """
    mirror joints
    Args:
        all_joints: joints
        num_joints: number of joints
    """
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res


def extend_crop(crop, crop_pad, image_size):
    """
    extend crop
    """
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop


def data_to_input(data):
    """
    transpose data to (C,H,W)
    """
    return np.transpose(data, [2, 0, 1]).astype(np.float32)


def get_pairwise_index(j_id, j_id_end, num_joints):
    """
    get pairwise index
    """
    return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)


def collect_pairwise_stats(joint_id, coords):
    """
    collect pairwise stats
    """
    pairwise_stats = {}
    for person_id in range(len(coords)):
        num_joints = len(joint_id[person_id])
        for k_start in range(num_joints):
            j_id_start = joint_id[person_id][k_start]
            joint_pt = coords[person_id][k_start, :]
            j_x_start = np.asscalar(joint_pt[0])
            j_y_start = np.asscalar(joint_pt[1])
            for k_end in range(num_joints):
                if k_start != k_end:
                    j_id_end = joint_id[person_id][k_end]
                    joint_pt = coords[person_id][k_end, :]
                    j_x_end = np.asscalar(joint_pt[0])
                    j_y_end = np.asscalar(joint_pt[1])
                    if (j_id_start, j_id_end) not in pairwise_stats:
                        pairwise_stats[(j_id_start, j_id_end)] = []
                    pairwise_stats[(j_id_start, j_id_end)].append([j_x_end - j_x_start, j_y_end - j_y_start])
    return pairwise_stats


def load_pairwise_stats(cfg):
    """
    load pairwise stats
    """
    mat_stats = sio.loadmat(cfg.pairwise_stats_fn)
    pairwise_stats = {}
    for _id in range(len(mat_stats['graph'])):
        pair = tuple(mat_stats['graph'][_id])
        pairwise_stats[pair] = {"mean": mat_stats['means'][_id], "std": mat_stats['std_devs'][_id]}
    for pair in pairwise_stats:
        pairwise_stats[pair]["mean"] *= cfg.global_scale
        pairwise_stats[pair]["std"] *= cfg.global_scale
    return pairwise_stats


class DataItem:
    pass


# noinspection PyAttributeOutsideInit
class PoseDataset:
    """
    basic dataset
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.dataset.type in [dataset.DATASET_TYPE_MPII_RAW,
                                dataset.DATASET_TYPE_COCO]:
            self.data = self.load_dataset() if cfg.dataset else []
        else:
            self.data = []
            return
        self.num_images = len(self.data)
        self.set_mirror(cfg.dataset.mirror)
        self.set_pairwise_stats_collect(cfg.pairwise_stats_collect)
        if self.cfg.pairwise_predict:
            self.pairwise_stats = load_pairwise_stats(self.cfg)

    def load_dataset(self):
        """
        load dataset
        """
        cfg = self.cfg
        file_name = cfg.dataset.path
        if cfg.context.device_target == "Ascend" and cfg.train:
            if self.cfg.model_arts.IS_MODEL_ARTS:
                file_name = self.cfg.model_arts.CACHE_INPUT + \
                    'crop/' + 'cropped/' + 'dataset.json'
            else:
                file_name = self.cfg.under_line.DATASET_ROOT + \
                  'mpii/' + 'cropped/' + 'dataset.json'

        # Load Matlab file dataset annotation
        with open(file_name, 'r') as f:
            mlab = f.read()
            mlab = json.loads(mlab)

        data = []
        has_gt = True

        for i, sample in enumerate(mlab):

            item = DataItem()
            item.image_id = i
            item.im_path = os.path.expanduser(sample['image'])
            item.im_size = np.array(sample['size'], dtype=np.int32)
            item.scale = self.get_scale()
            if not self.is_valid_size(item.im_size, item.scale):
                continue
            if sample.get('joints', None) is not None:
                joints = np.array(sample['joints'][0])
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert (joint_id < cfg.num_joints).any()
                joints[:, 0] = joint_id
                item.joints = [joints]
            else:
                has_gt = False
            if cfg.crop:
                crop = sample[3][0] - 1
                item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        self.has_gt = has_gt
        return data

    def set_mirror(self, mirror):
        """
        setup mirror
        """
        self.mirror = mirror
        if mirror:
            image_indices = np.arange(self.num_images * 2)
            self.mirrored = image_indices >= self.num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - self.num_images
            self.image_indices = image_indices
            self.symmetric_joints = mirror_joints_map(self.cfg.all_joints, self.cfg.num_joints)
        else:
            # assert not self.cfg.mirror
            self.image_indices = np.arange(self.num_images)
            self.mirrored = [False] * self.num_images

    def set_pairwise_stats_collect(self, pairwise_stats_collect):
        """
        setup pairwise stats collect
        """
        self.pairwise_stats_collect = pairwise_stats_collect
        if self.pairwise_stats_collect:
            assert self.get_scale() == 1.0

    def mirror_joint_coords(self, joints, image_width):
        """
        mirror joint coords
        horizontally flip the x-coordinate, keep y unchanged
        """
        # horizontally flip the x-coordinate, keep y unchanged
        joints[:, 1] = image_width - joints[:, 1] - 1
        return joints

    def mirror_joints(self, joints, symmetric_joints, image_width):
        """
        mirror joints
        """
        # joint ids are 0 indexed
        res = np.copy(joints)
        res = self.mirror_joint_coords(res, image_width)
        # swap the joint_id for a symmetric one
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res

    def num_training_samples(self):
        """
        the number of training samples
        """
        num = self.num_images
        if self.mirror:
            num *= 2
        return num

    def __len__(self):
        """
        the number of training samples
        """
        return self.num_training_samples()

    def __getitem__(self, item):
        """
        get a sample
        """
        batch = self.get_item(item)
        res = (
            batch.get(Batch.inputs, None),
            batch.get(Batch.part_score_targets, None),
            batch.get(Batch.part_score_weights, None),
            batch.get(Batch.locref_targets, None),
            batch.get(Batch.locref_mask, None),
        )
        if not self.cfg.train:
            res = (batch[Batch.data_item].im_path,) + res
        return res

    def get_idx_mirror(self, idx):
        """
        get real index and mirror
        """
        img_idx = self.image_indices[idx]
        mirror = self.cfg.dataset.mirror and self.mirrored[idx]

        return img_idx, mirror

    def get_training_sample(self, img_idx):
        """
        get sample
        """
        return self.data[img_idx]

    def get_scale(self):
        """
        get scale
        """
        cfg = self.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale

    def get_item(self, item):
        """
        get item
        """
        imidx, mirror = self.get_idx_mirror(item)
        data_item = self.get_training_sample(imidx)
        scale = data_item.scale

        return self.make_sample(data_item, scale, mirror)

    def is_valid_size(self, image_size, scale):
        """
        check image size
        """
        im_width = image_size[2]
        im_height = image_size[1]

        min_input_size = 100
        if im_height < min_input_size or im_width < min_input_size:
            return False

        if hasattr(self.cfg, 'max_input_size'):
            max_input_size = self.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height > max_input_size or input_width > max_input_size:
                return False

        return True

    def make_sample(self, data_item, scale, mirror):
        """
        make sample
        """
        cfg = self.cfg
        im_file = data_item.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)
        if cfg.context.device_target == "Ascend":
            if self.cfg.model_arts.IS_MODEL_ARTS:
                im_file = self.cfg.model_arts.CACHE_INPUT + \
                    'crop' + '/cropped/' + im_file.split('/')[-1]
            else:
                im_file = self.cfg.under_line.DATASET_ROOT + im_file
        image = imread(im_file, pilmode='RGB')

        if self.has_gt:
            joints = np.copy(data_item.joints)

        if self.cfg.crop:
            crop = data_item.crop
            image = image[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1, :]
            if self.has_gt:
                joints[:, 1:3] -= crop[0:2].astype(joints.dtype)
        image = Image.fromarray(image)
        if self.cfg.dataset.padding:
            new_image = Image.new("RGB", (self.cfg.max_input_size, self.cfg.max_input_size), (0, 0, 0))
            new_image.paste(image)
            image = new_image
        img = np.array(image.resize(
            (int(image.size[0] * scale), int(image.size[1] * scale))) if scale != 1 else image)
        image = np.array(image)
        scaled_img_size = arr(img.shape[0:2])

        if mirror:
            img = np.fliplr(img)

        batch = {Batch.inputs: img}

        if self.has_gt:
            stride = self.cfg.stride

            if mirror:
                joints = [self.mirror_joints(person_joints, self.symmetric_joints, image.shape[1]) for person_joints in
                          joints]

            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2

            scaled_joints = [person_joints[:, 1:3] * scale for person_joints in joints]

            joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
            batch = self.compute_targets_and_weights(joint_id, scaled_joints, data_item, sm_size, scale, batch)

            if self.pairwise_stats_collect:
                data_item.pairwise_stats = collect_pairwise_stats(joint_id, scaled_joints)

        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch

    def set_locref(self, locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy):
        """
        set location ref
        """
        locref_mask[j, i, j_id * 2 + 0] = 1
        locref_mask[j, i, j_id * 2 + 1] = 1
        locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
        locref_map[j, i, j_id * 2 + 1] = dy * locref_scale

    def set_pairwise_map(self, pairwise_map, pairwise_mask, i, j, j_id, j_id_end, coords, pt_x, pt_y, person_id, k_end):
        """
        set pairwise map
        """
        num_joints = self.cfg.num_joints
        joint_pt = coords[person_id][k_end, :]
        j_x_end = np.asscalar(joint_pt[0])
        j_y_end = np.asscalar(joint_pt[1])
        pair_id = get_pairwise_index(j_id, j_id_end, num_joints)
        stats = self.pairwise_stats[(j_id, j_id_end)]
        dx = j_x_end - pt_x
        dy = j_y_end - pt_y
        pairwise_mask[j, i, pair_id * 2 + 0] = 1
        pairwise_mask[j, i, pair_id * 2 + 1] = 1
        pairwise_map[j, i, pair_id * 2 + 0] = (dx - stats["mean"][0]) / stats["std"][0]
        pairwise_map[j, i, pair_id * 2 + 1] = (dy - stats["mean"][1]) / stats["std"][1]

    def compute_targets_and_weights(self, joint_id, coords, data_item, size, scale, batch):
        """
        compute targets and weights
        """
        stride = self.cfg.stride
        dist_thresh = self.cfg.pos_dist_thresh * scale
        num_joints = self.cfg.num_joints
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))

        locref_shape = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_shape)
        locref_map = np.zeros(locref_shape)

        pairwise_shape = cat([size, arr([num_joints * (num_joints - 1) * 2])])
        pairwise_mask = np.zeros(pairwise_shape)
        pairwise_map = np.zeros(pairwise_shape)

        dist_thresh_sq = dist_thresh ** 2

        width = size[1]
        height = size[0]

        self.loop_person(joint_id, coords, half_stride, stride, dist_thresh, width,
                         height, dist_thresh_sq,
                         locref_map,
                         scmap,
                         locref_mask, pairwise_map, pairwise_mask)
        scmap_weights = self.compute_score_map_weights(scmap.shape, joint_id, data_item)

        # Update batch
        batch.update({
            Batch.part_score_targets: scmap,
            Batch.part_score_weights: scmap_weights
        })
        if self.cfg.location_refinement:
            batch.update({
                Batch.locref_targets: locref_map,
                Batch.locref_mask: locref_mask
            })
        if self.cfg.pairwise_predict:
            batch.update({
                Batch.pairwise_targets: pairwise_map,
                Batch.pairwise_mask: pairwise_mask
            })

        return batch

    def loop_person(self, joint_id, coords, half_stride, stride, dist_thresh, width,
                    height, dist_thresh_sq,
                    locref_map,
                    scmap,
                    locref_mask, pairwise_map, pairwise_mask):
        for person_id in range(len(coords)):
            self.loop_joint(joint_id, person_id, coords, half_stride, stride, dist_thresh, width,
                            height, dist_thresh_sq,
                            locref_map,
                            scmap,
                            locref_mask, pairwise_map, pairwise_mask)

    def loop_joint(self, joint_id, person_id, coords, half_stride, stride, dist_thresh, width,
                   height, dist_thresh_sq,
                   locref_map,
                   scmap,
                   locref_mask, pairwise_map, pairwise_mask):
        for k, j_id in enumerate(joint_id[person_id]):
            joint_pt = coords[person_id][k, :]
            j_x = np.asscalar(joint_pt[0])
            j_y = np.asscalar(joint_pt[1])

            # don't loop over entire heatmap, but just relevant locations
            j_x_sm = round((j_x - half_stride) / stride)
            j_y_sm = round((j_y - half_stride) / stride)
            min_x = round(max(j_x_sm - dist_thresh - 1, 0))
            max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
            min_y = round(max(j_y_sm - dist_thresh - 1, 0))
            max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

            self.loop_area(stride, half_stride, min_x, max_x, min_y, max_y, j_x, j_y, dist_thresh_sq, locref_map,
                           j_id,
                           scmap,
                           locref_mask, joint_id, person_id, pairwise_map, pairwise_mask, coords, k)

    def loop_area(self, stride, half_stride, min_x, max_x, min_y, max_y, j_x, j_y, dist_thresh_sq, locref_map, j_id,
                  scmap,
                  locref_mask, joint_id, person_id, pairwise_map, pairwise_mask, coords, k):
        for j in range(min_y, max_y + 1):  # range(height):
            pt_y = j * stride + half_stride
            for i in range(min_x, max_x + 1):  # range(width):
                # pt = arr([i*stride+half_stride, j*stride+half_stride])
                # diff = joint_pt - pt
                # The code above is too slow in python
                pt_x = i * stride + half_stride
                # print(la.norm(diff))

                self.set_score_map(pt_x, pt_y, j_x, j_y, dist_thresh_sq, locref_map, i, j, j_id, scmap,
                                   locref_mask, joint_id, person_id, pairwise_map, pairwise_mask, coords, k)

    def set_score_map(self, pt_x, pt_y, j_x, j_y, dist_thresh_sq, locref_map, i, j, j_id, scmap, locref_mask,
                      joint_id,
                      person_id, pairwise_map, pairwise_mask, coords, k):

        dx = j_x - pt_x
        dy = j_y - pt_y
        dist = dx ** 2 + dy ** 2

        if dist <= dist_thresh_sq:
            dist = dx ** 2 + dy ** 2
            locref_scale = 1.0 / self.cfg.locref_stdev
            current_normalized_dist = dist * locref_scale ** 2
            prev_normalized_dist = locref_map[j, i, j_id * 2 + 0] ** 2 + locref_map[j, i, j_id * 2 + 1] ** 2
            update_scores = (scmap[j, i, j_id] == 0) or prev_normalized_dist > current_normalized_dist
            if self.cfg.location_refinement and update_scores:
                self.set_locref(locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy)
            if self.cfg.pairwise_predict and update_scores:
                for k_end, j_id_end in enumerate(joint_id[person_id]):
                    if k != k_end:
                        self.set_pairwise_map(pairwise_map, pairwise_mask, i, j, j_id, j_id_end,
                                              coords, pt_x, pt_y, person_id, k_end)
            scmap[j, i, j_id] = 1

    def compute_score_map_weights(self, scmap_shape, joint_id, data_item):
        """
        compute score map weights
        """
        cfg = self.cfg
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights
