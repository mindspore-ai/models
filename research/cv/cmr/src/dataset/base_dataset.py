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


from __future__ import division

from os.path import join
import numpy as np
import cv2

from src.utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from src import config as cfg


class BaseDataset:
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in config.py.
    """

    def __init__(self, options, dataset, use_augmentation=True, is_train=True):
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = cfg.DATASET_FOLDERS[dataset]
        self.data = np.load(cfg.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass

        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float32)
            self.betas = self.data['shape'].astype(np.float32)
            self.has_smpl = 1
        except KeyError:
            self.has_smpl = 0

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = np.array(1).astype(np.int32)
        except KeyError:
            self.has_pose_3d = np.array(0).astype(np.int32)

        # Get 2D keypoints
        try:
            self.keypoints = self.data['part']
        except KeyError:
            self.keypoints = np.zeros((len(self.imgname), 24, 3))

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2 * self.options.rot_factor,
                      max(-2 * self.options.rot_factor, np.random.randn() * self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.scale_factor,
                     max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [self.options.img_res, self.options.img_res], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0]*pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1]*pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = rgb_img.astype(np.float32) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2]+1, center, scale,
                                   [self.options.img_res, self.options.img_res], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1]/self.options.img_res - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S = np.einsum('ij,kj->ki', rot_mat, S)
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, str(self.imgname[index]))
        try:
            img = cv2.imread(imgname)[:, :, ::-1].astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)

        pose = self.pose_processing(pose, rot, flip)
        betas = betas.astype(np.float32)

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            St = self.j3d_processing(S.copy()[:, :-1], rot, flip)
            S[:, :-1] = St

            pose_3d = S
        else:
            pose_3d = np.zeros([24, 4])
        pose_3d = pose_3d.astype(np.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        keypoints = self.j2d_processing(keypoints, center, sc*scale, rot, flip)

        has_smpl = self.has_smpl
        has_pose_3d = self.has_pose_3d
        scale = float(sc * scale)
        center = center.astype(np.float32)

        # Pass path to segmentation mask, if available
        # Cannot load the mask because each mask has different size, so they cannot be stacked in one tensor
        try:
            maskname = self.maskname[index]
        except AttributeError:
            maskname = ''

        try:
            partname = self.partname[index]
        except AttributeError:
            partname = ''

        if self.is_train:
            return img, pose, betas, pose_3d, keypoints, has_smpl, has_pose_3d
        return img, pose, betas, center, scale, orig_shape, maskname, partname

    def __len__(self):
        return self.length
