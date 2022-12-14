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
import numpy as np
import h5py
import cv2
from src.config import config
from src.util import calc_temp_ab2, cut_image
from mindspore import set_seed
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision
set_seed(1234)


class Hum36mDataloaderP2:
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

    def _load_Dataset(self):
        self.images = []
        self.kp2ds = []
        self.boxs = []
        self.kp3ds = []
        self.shapes = []
        self.poses = []

        print('start loading hum3.6m data.')

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
                if '_3_' in total_image_names[index].decode():
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

        print('finished load hum3.6m data, total {} samples'.format(len(self.kp3ds)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        kp_3d = self.kp3ds[index].copy()
        np.random.seed(1234)
        scale = np.random.rand(
            4) * (self.scale_change[1] - self.scale_change[0]) + self.scale_change[0]
        originImage = cv2.imread(image_path)
        originImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
        image, kps = cut_image(originImage, kps, scale, box[0], box[1])
        ratio = 1.0 * config.crop_size / image.shape[0]
        kps[:, :2] *= ratio

        trivial, shape, pose = np.zeros(
            3), self.shapes[index], self.poses[index]
        theta = np.concatenate((trivial, pose, shape), 0)
        ratio = 1.0 / config.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        dst_image = cv2.resize(
            image,
            (config.crop_size,
             config.crop_size),
            interpolation=cv2.INTER_CUBIC)
        dst_image = self.Tensor(dst_image)
        data_ = {
            'kp_2d': kps,
            'kp_3d': kp_3d,
            'theta': theta,
            'image_name': self.images[index],
            'w_smpl': np.array([1.0]),
            'w_3d': np.array([1.0]),
            'data_set': 'hum3.6m'}

        label = np.concatenate(
            (data_['kp_2d'].flatten(),
             data_['kp_3d'].flatten(),
             data_['theta'],
             data_['w_smpl'],
             data_['w_3d']),
            axis=0).astype(
                np.float32)
        return dst_image, label


if __name__ == '__main__':
    DatasetHum = Hum36mDataloaderP2(
        dataset_path=config.dataset_path['hum3.6m'],
        is_crop=True,
        scale_change=[1.1, 1.2],
        is_flip=True,
        minpoints=5,
        pixelformat='NCHW',
        Normalization=True,
        pro_flip=0.5,
    )

    data = ds.GeneratorDataset(DatasetHum, ["data", "label"],
                               shuffle=False,
                               )
    dataset = data.batch(drop_remainder=True,
                         batch_size=1,
                         num_parallel_workers=config.num_worker,
                         python_multiprocessing=False)
    img_path = os.path.join(config.output_path, "img_data")
    label_path = os.path.join(config.output_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)
    for idx, data in enumerate(
            dataset.create_dict_iterator(
                output_numpy=True, num_epochs=1)):
        img_data = data["data"]
        img_label = data["label"]
        file_name = str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)
        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)
        print(idx)
    print("=" * 20, "export bin files finished", "=" * 20)
