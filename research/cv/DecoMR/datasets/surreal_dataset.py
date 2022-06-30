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

import os
from os.path import join
import numpy as np
from utils import config as cfg
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import mindspore.dataset.vision.c_transforms as c_vision
import cv2


def flip_smpl_kp(kp):
    """Flip SMPL 24 keypoints."""
    flipped_parts = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    kp = kp[flipped_parts]
    kp[:, 0] = - kp[:, 0]
    return kp

class SurrealDataset:
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """
    def __init__(self, options, use_augmentation=True, is_train=True, use_IUV=False):
        super(SurrealDataset, self).__init__()
        all_dataset = 'surreal'

        self.is_train = is_train
        self.options = options
        root_dir = cfg.DATASET_FOLDERS[all_dataset]
        if self.is_train:
            self.video_dir = join(root_dir, 'cmu/train')
        else:
            self.video_dir = join(root_dir, 'cmu/val')

        self.normalize_img = c_vision.Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
        self.data = np.load(cfg.DATASET_FILES[is_train][all_dataset])

        self.videoname = self.data['videoname']
        self.framenum = self.data['framenum']
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.use_augmentation = use_augmentation

        # Get gender data, if available
        try:
            self.gender = self.data['gender']
            #self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(float)
            self.betas = self.data['shape'].astype(float)
            self.has_smpl = np.ones(len(self.imgname)).astype(int)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname)).astype(int)

        # Get gt 3D pose, if available
        try:
            self.pose_3d_smpl = self.data['S_smpl']
            self.has_pose_3d_smpl = 1
        except KeyError:
            self.has_pose_3d_smpl = 0

        # Get 2D keypoints
        try:
            self.keypoints_smpl = self.data['part_smpl']
        except KeyError:
            self.keypoints_smpl = np.zeros((len(self.scale), 24, 3))

        self.length = self.scale.shape[0]

        self.use_IUV = use_IUV
        self.has_dp = np.zeros(len(self.scale))

        if self.use_IUV and is_train:
            self.iuvname = self.data['iuv_names']
            self.has_dp = self.has_smpl
            self.uv_type = options.uv_type
            self.iuv_dir = join(root_dir, '{}_IUV_gt'.format(self.uv_type), 'cmu/train')


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            if self.options.use_augmentation:
                # We flip with probability 1/2
                if np.random.uniform() <= 0.5:
                    flip = 1

                # Each channel is multiplied with a number
                # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
                pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)

                # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
                rot = min(2*self.options.rot_factor,
                          max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))

                # The scale is multiplied with a number
                # in the area [1-scaleFactor,1+scaleFactor]
                sc = min(1+self.options.scale_factor,
                         max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
                # but it is zero with probability 3/5
                if np.random.uniform() <= 0.6:
                    rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [self.options.img_res, self.options.img_res], rot=rot)

        if flip:
            rgb_img = flip_img(rgb_img)

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

        rgb_img = rgb_img / 255.0
        return rgb_img

    def smpl_j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [self.options.img_res, self.options.img_res], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.img_res - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def smpl_j3d_processing(self, S, r, f):
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
            S = flip_smpl_kp(S)
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

    def iuv_processing(self, IUV, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        IUV = crop(IUV, center, scale,
                   [self.options.img_res, self.options.img_res], rot=rot)

        if flip:
            IUV = flip_img(IUV)
            if self.uv_type == 'BF':
                mask = (IUV[0] > 0).astype('float32')
                IUV[1] = (255 - IUV[1]) * mask
            else:
                print('Flip augomentation for SMPL default UV map is not supported yet.')
        IUV = np.transpose(IUV.astype('float32'), (2, 0, 1))
        return IUV

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Extract the frame from the video
        videoname = self.videoname[index]
        frame = self.framenum[index]
        cap = cv2.VideoCapture(join(self.video_dir, videoname))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        # The image should be mirrored first
        img = np.fliplr(img)[:, :, ::-1].copy().astype(np.float32)
        orig_shape = np.array(img.shape)[:2]
        item['scale'] = (sc * scale).astype('float32')
        item['center'] = center.astype('float32')
        item['orig_shape'] = orig_shape.astype('int32')

        # Process image
        img = self.rgb_processing(img, center, sc * scale, rot, flip, pn).astype('float32')
        # Store image before normalization to use it in visualization
        item['img_orig'] = img.copy()
        item['img'] = np.transpose(self.normalize_img(img).astype('float32'), (2, 0, 1))
        item['imgname'] = videoname + '_frame_{}'.format('frame')

        # Get SMPL parameters, if available
        has_smpl = self.has_smpl[index]
        item['has_smpl'] = has_smpl
        if has_smpl:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
        item['pose'] = self.pose_processing(pose, rot, flip)
        item['betas'] = betas.astype('float32')

        # Surreal dataset does NOT provide the ground truth of keypoints.
        # The keypoints of SURREAL is the 24 joints defined by SMPL model.
        item['keypoints'] = np.zeros((24, 3)).astype('float32')
        item['pose_3d'] = np.zeros((24, 4)).astype('float32')
        item['has_pose_3d'] = 0

        # Get 3D and 2D GT SMPL joints (For the compatibility with SURREAL dataset)
        item['has_pose_3d_smpl'] = self.has_pose_3d_smpl
        if self.has_pose_3d_smpl:
            S = self.pose_3d_smpl[index].copy()
            St = self.smpl_j3d_processing(S.copy()[:, :-1], rot, flip)
            S[:, :-1] = St
            item['pose_3d_smpl'] = S.astype('float32')
        else:
            item['pose_3d_smpl'] = np.zeros((24, 4)).astype('float32')

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints_smpl[index].copy()
        item['keypoints_smpl'] = self.j2d_processing(keypoints, center, sc * scale, rot, flip)

        # Pass path to segmentation mask, if available
        # Cannot load the mask because each mask has different size, so they cannot be stacked in one tensor
        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''
        item['gender'] = self.gender[index]

        if self.use_IUV:
            IUV = np.zeros((3, img.shape[1], img.shape[2])).astype('float32')
            iuvname = ''
            has_dp = self.has_dp[index]
            try:
                fit_error = self.fit_joint_error[index]
            except AttributeError:
                fit_error = 0.0         # For the dataset with GT mesh, fit_error is set 0

            if has_dp:
                iuvname = join(self.iuv_dir, str(self.iuvname[index]))
                if os.path.exists(iuvname):
                    IUV = cv2.imread(iuvname).copy()
                    IUV = self.iuv_processing(IUV, center, sc * scale, rot, flip, pn)  # process UV map
                else:
                    has_dp = 0
                    print("GT IUV image: {} does not exist".format(iuvname))

            item['gt_iuv'] = IUV
            item['iuvname'] = iuvname
            item['has_dp'] = has_dp
            item['fit_joint_error'] = fit_error

        if self.use_IUV:
            return item['scale'], item['center'], item['orig_shape'], item['img_orig'], item['img'],\
                   item['imgname'], item['has_smpl'], item['pose'], item['betas'], item['has_pose_3d'],\
                   item['pose_3d'], item['keypoints'], item['keypoints_smpl'], item['pose_3d_smpl'], \
                   item['has_pose_3d_smpl'], item['maskname'], item['partname'], item['gender'], item['gt_iuv'],\
                   item['iuvname'], item['has_dp'], item['fit_joint_error']

        return item['scale'], item['center'], item['orig_shape'], item['img_orig'], item['img'], \
               item['imgname'], item['has_smpl'], item['pose'], item['betas'], item['has_pose_3d'], \
               item['pose_3d'], item['keypoints'], item['keypoints_smpl'], item['pose_3d_smpl'], \
               item['has_pose_3d_smpl'], item['maskname'], item['partname'], item['gender']

    def __len__(self):
        return len(self.imgname)
