
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
"""
Add notes.
"""
import sys
import json
import os
import glob
import random
import h5py
import numpy as np
import scipy.io as scio
import cv2
from src.util import calc_temp_ab2, cut_image, flip_image, reflect_pose, reflect_lsp_kp, get_rectangle_intersect_ratio
from src.config import config
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision


class Hum36mDataloader:
    def __init__(
            self,
            dataset_path,
            is_crop,
            scale_change,
            is_flip,
            minpoints,
            pixelformat='NHWC',
            Normalization=False,
            pro_flip=0.3):
        self.data_folder = dataset_path
        self.is_crop = is_crop
        self.scale_change = scale_change
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.minpoints = minpoints
        self.pixelformat = pixelformat
        self.Normalization = Normalization
        self.Tensor = py_vision.ToTensor()
        self._load_Dataset()
        print('finished load hum3.6m data')

    def _load_Dataset(self):

        self.images = []
        self.kp2ds = []
        self.boxs = []
        self.kp3ds = []
        self.shapes = []
        self.poses = []

        anno_file_path = os.path.join(self.data_folder, 'annot.h5')
        with h5py.File(anno_file_path) as fp:
            total_kp2d = np.array(fp['gt2d'])
            total_kp3d = np.array(fp['gt3d'])
            total_shap = np.array(fp['shape'])
            total_pose = np.array(fp['pose'])
            total_image_names = np.array(fp['imagename'])

            assert len(total_kp2d) == len(total_kp3d) and len(total_kp2d) == len(total_image_names) and \
                len(total_kp2d) == len(total_shap) and len(total_kp2d) == len(total_pose)

            l = len(total_kp2d)

            def _collect_valid_pts(pts):
                r = []
                for pt in pts:
                    if pt[2] != 0:
                        r.append(pt)
                return r

            for index in range(l):
                if 'S9' not in total_image_names[index].decode(
                ) and 'S11' not in total_image_names[index].decode():
                    kp2d = total_kp2d[index]
                    if np.sum(kp2d[:, 2]) < self.minpoints:
                        continue

                    lt, rb, _ = calc_temp_ab2(_collect_valid_pts(kp2d))
                    self.kp2ds.append(
                        np.array(kp2d.copy().reshape(-1, 3), dtype=np.float))
                    self.boxs.append((lt, rb))
                    self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                    self.shapes.append(total_shap[index].copy())
                    self.poses.append(np.sum(total_pose[index].copy(), axis=0))
                    self.images.append(
                        os.path.join(
                            self.data_folder,
                            'images',
                            total_image_names[index].decode()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        kp_3d = self.kp3ds[index].copy()

        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])

        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)

        trivial, shape, pose = np.zeros(
            3), self.shapes[index], self.poses[index]
        if self.is_flip:
            dst_image, kps = flip_image(dst_image, kps)
            pose = reflect_pose(pose)
            kp_3d = reflect_lsp_kp(kp_3d)

        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0

        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)

        dst_image = self.Tensor(dst_image)
        theta = np.concatenate((trivial, pose, shape), 0)

        label = np.concatenate(
            (kps.flatten(),
             kp_3d.flatten(),
             theta,
             np.array([1.0]),
             np.array([1.0])),
            axis=0).astype(
                np.float32)
        return dst_image, label


class COCO2017Dataloader:

    def __init__(
            self,
            dataset_path,
            is_crop,
            scale_change,
            is_flip,
            only_single_person,
            minpoints,
            max_intersec_ratio=0.1,
            pixelformat='NHWC',
            Normalization=False,
            pro_flip=0.3):
        self.data_folder = dataset_path
        self.is_crop = is_crop
        self.scale_change = scale_change
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.only_single_person = only_single_person
        self.minpoints = minpoints
        self.max_intersec_ratio = max_intersec_ratio
        self.pixelformat = pixelformat
        self.Normalization = Normalization
        self._load_Dataset()
        self.data = None
        self.label = None
        self.Tensor = py_vision.ToTensor()
        print('finished load coco data')

    def _load_Dataset(self):
        self.images = []
        self.kp2ds = []
        self.boxs = []
        anno_file_path = os.path.join(
            self.data_folder,
            'annotations',
            'person_keypoints_train2017.json')
        with open(anno_file_path, 'r') as reader:
            anno = json.load(reader)

        file_list_train = os.listdir(
            os.path.join(
                self.data_folder,
                'train2017'))

        def _image_ID(imageid_info, images_INFO):
            for image_info in images_INFO:
                image_id = image_info['id']
                image_name = image_info['file_name']
                _anno = {}
                if image_name in file_list_train:
                    _anno['image_path'] = os.path.join(
                        self.data_folder, 'train2017', image_name)
                else:
                    _anno['image_path'] = os.path.join(
                        self.data_folder, 'val2017', image_name)
                _anno['kps'] = []
                _anno['box'] = []
                imageid_info[image_id] = _anno

        images = anno['images']

        imageid_info = {}
        _image_ID(imageid_info, images)

        annos = anno['annotations']
        for anno_info in annos:
            self._deal_annoinfo(anno_info, imageid_info)

        for _, v in imageid_info.items():
            self._deal_imageinfo(v)

    def _deal_imageinfo(self, image_info):
        image_path = image_info['image_path']
        kp_set = image_info['kps']
        box_set = image_info['box']
        if len(box_set) > 1:
            if self.only_single_person:
                return

        for _ in range(len(box_set)):
            self._deal_sample(_, kp_set, box_set, image_path)

    def _deal_sample(self, key, kps, boxs, image_path):
        def _collect_box(l, boxs):
            r = []
            for _ in range(len(boxs)):
                if _ == l:
                    continue
                r.append(boxs[_])
            return r

        def _collide_heavily(box, boxs):
            for it in boxs:
                if get_rectangle_intersect_ratio(
                        box[0], box[1], it[0], it[1]) > self.max_intersec_ratio:
                    return True
            return False

        kp = kps[key]
        box = boxs[key]

        valid_pt_cound = np.sum(kp[:, 2])
        if valid_pt_cound < self.minpoints:
            return

        r = _collect_box(key, boxs)
        if _collide_heavily(box, r):
            return
        self.images.append(image_path)
        self.kp2ds.append(kp.copy())
        self.boxs.append(box.copy())

    def _deal_annoinfo(self, anno_info, imageid_info):

        image_id = anno_info['image_id']
        kps = anno_info['keypoints']
        box_info = anno_info['bbox']
        box = [np.array([int(box_info[0]), int(box_info[1])]), np.array(
            [int(box_info[0] + box_info[2]), int(box_info[1] + box_info[3])])]
        assert image_id in imageid_info
        _anno = imageid_info[image_id]
        _anno['box'].append(box)
        _anno['kps'].append(self._convert_14_pts(kps))

    def _convert_14_pts(self, coco_pts):
        kp_map = [15, 13, 11, 10, 12, 14, 9, 7, 5, 4, 6, 8, 0, 0]
        kp_map = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 0, 0]
        kps = np.array(coco_pts, dtype=np.float).reshape(-1, 3)[kp_map].copy()
        kps[12:, 2] = 0.0  # no neck, top head
        kps[:, 2] /= 2.0
        return kps

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])
        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)

        if self.is_flip and random.random() <= self.pro_flip:
            dst_image, kps = flip_image(dst_image, kps)

        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0

        label = kps.flatten()

        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)
        dst_image = self.Tensor(dst_image)

        return dst_image, label.astype(np.float32)


class LspLoader:
    def __init__(
            self,
            dataset_path,
            is_crop,
            scale_change,
            is_flip,
            pixelformat='NHWC',
            Normalization=False,
            pro_flip=0.3):

        self.is_crop = is_crop
        self.scale_change = scale_change
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.data_folder = dataset_path
        self.pixelformat = pixelformat
        self.Normalization = Normalization
        self.Tensor = py_vision.ToTensor()
        self._load_Dataset()
        print('finished load lsp data')

    def __len__(self):
        return len(self.images)

    def _load_Dataset(self):
        self.images = []
        self.kp2ds = []
        self.boxs = []

        anno_file_path = os.path.join(self.data_folder, 'joints.mat')
        anno = scio.loadmat(anno_file_path)
        kp2d = anno['joints'].transpose(2, 1, 0)
        visible = np.logical_not(kp2d[:, :, 2])
        kp2d[:, :, 2] = visible.astype(kp2d.dtype)
        image_folder = os.path.join(self.data_folder, 'images')
        images = sorted(glob.glob(image_folder + '/im*.jpg'))
        for _ in range(len(images)):
            self._handle_image(images[_], kp2d[_])

    def _handle_image(self, image_path, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] == 1:
                pt_valid.append(pt)
        lt, rb, valid = calc_temp_ab2(pt_valid)

        if not valid:
            return

        self.kp2ds.append(kps.copy().astype(np.float))
        self.images.append(image_path)
        self.boxs.append((lt, rb))

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])
        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)
        if self.is_flip and random.random() <= self.pro_flip:
            dst_image, kps = flip_image(dst_image, kps)

        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        dst_image = self.Tensor(dst_image)
        label = kps.flatten().astype(np.float32)

        return dst_image, label


class LspExtLoader:
    def __init__(
            self,
            dataset_path,
            is_crop,
            scale_change,
            is_flip,
            pixelformat='NHWC',
            Normalization=False,
            pro_flip=0.3):
        self.is_crop = is_crop
        self.scale_change = scale_change
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.data_folder = dataset_path
        self.pixelformat = pixelformat
        self.Normalization = Normalization
        self.Tensor = py_vision.ToTensor()
        self._load_Dataset()
        self.data = None
        self.label = None

        self._load_Dataset()
        print('finished load lsp-exd data')

    def _load_Dataset(self):

        self.images = []
        self.kp2ds = []
        self.boxs = []

        anno_file_path = os.path.join(self.data_folder, 'joints.mat')
        anno = scio.loadmat(anno_file_path)
        kp2d = anno['joints'].transpose(2, 0, 1)  # N x k x 3
        image_folder = os.path.join(self.data_folder, 'images')
        images = sorted(glob.glob(image_folder + '/im*.jpg'))
        for _ in range(len(images)):
            self._handle_image(images[_], kp2d[_])

    def _handle_image(self, image_path, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] == 1:
                pt_valid.append(pt)
        lt, rb, valid = calc_temp_ab2(pt_valid)

        if not valid:
            return

        self.kp2ds.append(kps.copy().astype(np.float))
        self.images.append(image_path)
        self.boxs.append((lt, rb))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])
        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)

        if self.is_flip and random.random() <= self.pro_flip:
            dst_image, kps = flip_image(dst_image, kps)
        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        label = kps.flatten().astype(np.float32)
        dst_image = self.Tensor(dst_image)
        return dst_image, label


class MoshDataloader():
    def __init__(self, dataset_path, is_flip=True, pro_flip=0.3):

        self.data_folder = dataset_path
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.shapes = []
        self.poses = []
        self.data = None
        self._load_Dataset()
        print('finished load mosh data')

    def _load_Dataset(self):
        anno_file_path = os.path.join(self.data_folder, 'annot.h5')
        with h5py.File(anno_file_path) as fp:
            total_shap = np.array(fp['shape'])
            total_pose = np.array(fp['pose'])
        l = len(total_shap)
        for index in range(l):
            self.shapes.append(total_shap[index].copy())
            self.poses.append(np.sum(total_pose[index].copy(), axis=0))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        trivial, pose, shape = np.zeros(
            3), self.poses[index], self.shapes[index]

        data = np.concatenate(
            (trivial, pose.flatten(), shape.flatten()), axis=0)
        data = data.astype(np.float32)

        return data


class MpiInf3dhpDataloader():
    def __init__(
            self,
            dataset_path,
            is_crop,
            scale_change,
            is_flip,
            minpoints,
            pixelformat='NHWC',
            Normalization=False,
            pro_flip=0.3):
        self.data_folder = dataset_path
        self.is_crop = is_crop
        self.scale_change = scale_change
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.minpoints = minpoints
        self.pixelformat = pixelformat
        self.Normalization = Normalization
        self.data = None
        self.label = None
        self.Tensor = py_vision.ToTensor()
        self._load_Dataset()
        print('finished load mpi-inf-3d data')

    def _load_Dataset(self):
        self.images = []
        self.kp2ds = []
        self.boxs = []
        self.kp3ds = []

        annos_path = os.path.join(self.data_folder, 'annot.h5')
        with h5py.File(annos_path) as fp:
            kps_2d = np.array(fp['gt2d'])
            kps_3d = np.array(fp['gt3d'])
            image_Names = np.array(fp['imagename'])

            assert len(kps_2d) == len(kps_3d) and len(
                kps_2d) == len(image_Names)

            l = len(kps_2d)

            def _collect_valid_pts(pts):
                r = []
                for pt in pts:
                    if pt[2] != 0:
                        r.append(pt)
                return r

            for index in range(l):
                if 'S8' not in image_Names[index].decode():
                    kp2d = kps_2d[index].reshape((-1, 3))
                    if np.sum(kp2d[:, 2]) < self.minpoints:
                        continue

                    lt, rb, _ = calc_temp_ab2(_collect_valid_pts(kp2d))
                    self.kp2ds.append(
                        np.array(kp2d.copy().reshape(-1, 3), dtype=np.float))
                    self.boxs.append((lt, rb))
                    self.kp3ds.append(kps_3d[index].copy().reshape(-1, 3))
                    self.images.append(
                        os.path.join(
                            self.data_folder,
                            'images',
                            image_Names[index].decode()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        kp_3d = self.kp3ds[index].copy()

        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])

        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)

        if self.is_flip:
            dst_image, kps = flip_image(dst_image, kps)
            kp_3d = reflect_lsp_kp(kp_3d)

        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)
        dst_image = self.Tensor(dst_image)

        label = np.concatenate(
            (kps.flatten(),
             kp_3d.flatten(),
             np.zeros(85),
             np.array([1.0]),
             np.array([1.0])),
            axis=0).astype(
                np.float32)
        return dst_image, label


class MpiiLoader:

    def __init__(
            self,
            dataset_path,
            is_crop,
            is_flip=True,
            pro_flip=0.3):

        self.is_crop = is_crop
        self.scale_change = [1.05, 1.3]
        self.is_flip = is_flip
        self.pro_flip = pro_flip
        self.data_folder = dataset_path
        img_directory = dataset_path
        anno_mat = os.path.join(
            img_directory,
            'annotations',
            'mpii_human_pose_v1_u12_1.mat')
        anno = self.load_anno(anno_mat)
        img_dir = os.path.join(img_directory, 'images')
        self.process_mpii(anno, img_dir, is_train=True)
        self.Tensor = py_vision.ToTensor()
        self._load_Dataset()

    def _load_Dataset(self):
        print('loading Mpii data.')
        self.images = self.filename_
        self.kp2ds = []
        self.boxs = []
        kp2d = self.kps_
        visible = np.logical_not(kp2d[:, :, 2])
        kp2d[:, :, 2] = visible.astype(kp2d.dtype)

        images = self.filename_
        for _ in range(len(images)):
            self._handle_image(kp2d[_])

        print('finished load Mpii data.')

    def __len__(self):
        return len(self.images)

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            pt_valid.append(pt)
        lt, rb, valid = calc_temp_ab2(pt_valid)

        if not valid:
            return

        self.kp2ds.append(kps.copy().astype(np.float))
        self.boxs.append((lt, rb))

    def load_anno(self, fname):

        res = scio.loadmat(fname, struct_as_record=False, squeeze_me=True)

        return res['RELEASE']

    def convert_is_visible(self, is_visible):

        if isinstance(is_visible, np.ndarray):
            tmp = 0
        else:
            tmp = int(is_visible)
        return tmp

    def read_joints(self, rect):
        jointsIds = [
            0,  # R ankle
            1,  # R knee
            2,  # R hip
            3,  # L hip
            4,  # L knee
            5,  # L ankle
            10,  # R Wrist
            11,  # R Elbow
            12,  # R shoulder
            13,  # L shoulder
            14,  # L Elbow
            15,  # L Wrist
            8,  # Neck top
            9,  # Head top
        ]
        points = rect.annopoints.point
        if not isinstance(points, np.ndarray):
            return None
        read_points = {}

        for point in points:
            vis = self.convert_is_visible(point.is_visible)
            read_points[point.id] = np.array([point.x, point.y, vis])
        joints = np.zeros((3, len(jointsIds)))
        for i, jid in enumerate(jointsIds):
            if jid in read_points.keys():
                joints[:, i] = read_points[jid]
                joints[2, i] = 1.

        return joints

    def parse_people(self, anno_info, single_persons):

        if single_persons.size == 0:
            return []

        rects = anno_info.annorect
        if not isinstance(rects, np.ndarray):
            rects = np.array([rects])

        people = []

        for ridx in single_persons:
            rect = rects[ridx - 1]
            pos = np.array([rect.objpos.x, rect.objpos.y])
            joints = self.read_joints(rect)
            if joints is None:
                continue
            visible = joints[2, :].astype(bool)
            if visible[0] or visible[5]:
                min_pt = np.min(joints[:2, visible], axis=1)
                max_pt = np.max(joints[:2, visible], axis=1)
                person_height = np.linalg.norm(max_pt - min_pt)
                scale = 150. / person_height
            else:
                torso_heights = list()
                if visible[13] and visible[2]:
                    torso_heights.append(
                        np.linalg.norm(joints[:2, 13] - joints[:2, 2]))
                if visible[13] and visible[3]:
                    torso_heights.append(
                        np.linalg.norm(joints[:2, 13] - joints[:2, 3]))

                if torso_heights > 0:
                    scale = 75. / np.mean(torso_heights)
                else:
                    if visible[8] and visible[2]:
                        torso_heights.append(
                            np.linalg.norm(joints[:2, 8] - joints[:2, 2]))
                    if visible[9] and visible[3]:
                        torso_heights.append(
                            np.linalg.norm(joints[:2, 9] - joints[:2, 3]))
                    if torso_heights > 0:
                        scale = 56. / np.mean(torso_heights)
                    else:
                        continue
            people.append((joints, scale, pos))

        return people

    def add_to_tfrecord(self, anno, img_id):

        anno_info = anno.annolist[img_id]
        image_name = anno_info.image.name
        single_persons = anno.single_person[img_id]
        if not isinstance(single_persons, np.ndarray):
            single_persons = np.array([single_persons])

        people = self.parse_people(anno_info, single_persons)
        return len(people), image_name, people

    def process_mpii(self, anno, img_dir, is_train=True):
        all_ids = np.array(range(len(anno.annolist)))
        if is_train:

            img_inds = all_ids[anno.img_train.astype('bool')]
        else:
            img_inds = all_ids[np.logical_not(anno.img_train)]
            print('Not implemented for test data')
            exit(1)

        i = 0
        fidx = 0
        kps_ = []
        filename_ = []
        while i < len(img_inds):
            while i < len(img_inds):
                length_, filename, people = self.add_to_tfrecord(
                    anno, img_inds[i])
                if length_ == 1:
                    _ = np.transpose(people[0][0])
                    kps_.append(_)
                    filename_.append(os.path.join(img_dir, filename))
                i += 1

            fidx += 1
        self.kps_ = np.stack(kps_)
        self.filename_ = filename_

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])
        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)

        if self.is_flip and random.random() <= self.pro_flip:
            dst_image, kps = flip_image(dst_image, kps)
        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)
        dst_image = self.Tensor(dst_image)

        return dst_image, kps


class Data:

    def _create_data_loader(self):
        self.loader_2d = self._create_2d_data_loader(
            config.train_2d_set)
        self.loader_mosh = self._create_adv_data_loader(
            config.train_adv_set)
        self.loader_3d = self._create_3d_data_loader(
            config.train_3d_set)

    def _create_2d_data_loader(self, data_2d_set):
        for data_set_name in data_2d_set:
            dataset_path = config.dataset_path[data_set_name]
            if data_set_name == 'coco':
                coco = COCO2017Dataloader(dataset_path=dataset_path,
                                          is_crop=True,
                                          scale_change=[1.05, 1.3],
                                          is_flip=self.is_flip,
                                          only_single_person=False,
                                          minpoints=7,
                                          max_intersec_ratio=0.5,
                                          pixelformat=self.pixelformat,
                                          Normalization=self.Normalization,
                                          pro_flip=self.pro_flip)

                dataset_1 = ds.GeneratorDataset(
                    coco,
                    ["data", "label"],
                    shuffle=True,
                    num_shards=self.device_num,
                    shard_id=self.rank_id
                )
                dataset_1 = dataset_1.batch(
                    drop_remainder=True,
                    batch_size=config.batch_size,
                    num_parallel_workers=config.num_worker,
                )
            elif data_set_name == 'lsp':
                lsp = LspLoader(dataset_path=dataset_path,
                                is_crop=True,
                                scale_change=[1.05, 1.3],
                                is_flip=self.is_flip,
                                pixelformat=self.pixelformat,
                                Normalization=self.Normalization,
                                pro_flip=self.pro_flip)

                dataset_2 = ds.GeneratorDataset(
                    lsp,
                    ["data", "label"],
                    shuffle=True,
                    num_shards=self.device_num,
                    shard_id=self.rank_id
                )
                dataset_2 = dataset_2.batch(
                    drop_remainder=True,
                    batch_size=config.batch_size,
                    num_parallel_workers=config.num_worker,
                )
            elif data_set_name == 'lsp_ext':
                lsp_ext = LspExtLoader(dataset_path=dataset_path,
                                       is_crop=True,
                                       scale_change=[1.1, 1.2],
                                       is_flip=self.is_flip,
                                       pro_flip=self.pro_flip)
                dataset_3 = ds.GeneratorDataset(
                    lsp_ext,
                    ["data", "label"],
                    shuffle=True,
                    num_shards=self.device_num,
                    shard_id=self.rank_id
                )
                dataset_3 = dataset_3.batch(
                    drop_remainder=True,
                    batch_size=config.batch_size,
                    num_parallel_workers=config.num_worker,
                )

            else:
                msg = 'invalid 2d dataset'
                sys.exit(msg)

        data_tmp = dataset_1.concat(dataset_2)
        data_tmp = data_tmp.concat(dataset_3)

        return data_tmp

    def _create_3d_data_loader(self, data_3d_set):
        for data_set_name in data_3d_set:
            dataset_path = config.dataset_path[data_set_name]
            if data_set_name == 'mpi-inf-3dhp':
                mpi_inf_3dhp = MpiInf3dhpDataloader(
                    dataset_path=dataset_path,
                    is_crop=True,
                    scale_change=[1.1, 1.2],
                    is_flip=self.is_flip,
                    minpoints=5,
                    pixelformat=self.pixelformat,
                    Normalization=self.Normalization,
                    pro_flip=self.pro_flip)
                dataset_4 = ds.GeneratorDataset(
                    mpi_inf_3dhp,
                    ["data", "label"],
                    shuffle=True,
                    num_shards=self.device_num,
                    shard_id=self.rank_id
                )
                dataset_4 = dataset_4.batch(
                    drop_remainder=True,
                    batch_size=config.batch_3d_size,
                    num_parallel_workers=config.num_worker,
                )
            elif data_set_name == 'hum3.6m':
                hum36m = Hum36mDataloader(dataset_path=dataset_path,
                                          is_crop=True,
                                          scale_change=[1.1, 1.2],
                                          is_flip=self.is_flip,
                                          minpoints=5,
                                          pixelformat=self.pixelformat,
                                          Normalization=self.Normalization,
                                          pro_flip=self.pro_flip)
                dataset_5 = ds.GeneratorDataset(
                    hum36m,
                    ["data", "label"],
                    shuffle=True,
                    num_shards=self.device_num,
                    shard_id=self.rank_id
                )
                dataset_5 = dataset_5.batch(
                    drop_remainder=True,
                    batch_size=config.batch_3d_size,
                    num_parallel_workers=config.num_worker,
                )
            else:
                msg = 'dataset is error'
                sys.exit(msg)
        data_tmp = dataset_4.concat(dataset_5)
        return data_tmp

    def _create_adv_data_loader(self, data_adv_set):
        for data_set_name in data_adv_set:
            dataset_path = config.dataset_path[data_set_name]
            if data_set_name == 'mosh':
                mosh = MoshDataloader(dataset_path=dataset_path,
                                      is_flip=self.is_flip,
                                      pro_flip=self.pro_flip)
                dataset_6 = ds.GeneratorDataset(
                    mosh,
                    ["data"],
                    shuffle=True,
                    num_shards=self.device_num,
                    shard_id=self.rank_id
                )
                dataset_6 = dataset_6.batch(
                    drop_remainder=True,
                    batch_size=config.adv_batch_size,
                    num_parallel_workers=config.num_worker,
                )
            else:
                msg = 'dataset is error'
                sys.exit(msg)
        return dataset_6
