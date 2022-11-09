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
"""MVTecDataset"""
import glob
import json
import os
from pathlib import Path
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.transforms import Compose
from mindspore.dataset.vision import Inter
from PIL import Image


class MVTecDataset():
    """MVTecDataset"""
    def __init__(self, root, transform, gt_transform, phase, is_json=False):
        self.phase = phase
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.is_json = is_json
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        """load_dataset"""
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(self.transform(img)[0], dtype=np.float32)
        if self.phase == 'train':
            return img

        if gt == 0:
            gt = np.zeros((1, np.array(img).shape[-2], np.array(img).shape[-2]), dtype=np.float32)
        else:
            gt = Image.open(gt)
            gt = np.array(self.gt_transform(gt)[0], dtype=np.float32)

        if self.is_json:
            return os.path.basename(img_path[:-4]), img_type

        return img, gt, label, idx

def createDatasetJson(dataset_path, category, test_data, phase='test'):
    """createDatasetJson"""
    path = os.path.join(dataset_path, 'json', category)
    if not os.path.exists(path):
        os.makedirs(path)
    test_json_path = os.path.join(path, '{}_{}.json'.format(category, phase))

    if not os.path.isfile(test_json_path):
        os.mknod(test_json_path)
        test_label = {}
        test_data.is_json = True
        test_data_length = test_data.__len__()
        for i in range(test_data_length):
            single_label = {}
            name, img_type = test_data.__getitem__(i)
            single_label['name'] = name
            single_label['img_type'] = img_type
            test_label['{}'.format(i)] = single_label

        json_path = Path(test_json_path)
        with json_path.open('w') as json_file:
            json.dump(test_label, json_file)

    return test_json_path

def createDataset(dataset_path, category, train_batch_size=32, test_batch_size=1, im_resize=256):
    """createDataset"""
    # Computed from random subset of ImageNet training images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = Compose([
        vision.Resize((im_resize, im_resize), interpolation=Inter.BILINEAR),
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std, is_hwc=False)
    ])
    gt_transforms = Compose([
        vision.Resize((im_resize, im_resize), interpolation=Inter.BILINEAR),
        vision.ToTensor()
    ])

    train_data = MVTecDataset(root=os.path.join(dataset_path, category),
                              transform=data_transforms, gt_transform=gt_transforms, phase='train')
    test_data = MVTecDataset(root=os.path.join(dataset_path, category),
                             transform=data_transforms, gt_transform=gt_transforms, phase='test')

    train_dataset = ds.GeneratorDataset(train_data, column_names=['img'],
                                        shuffle=True, num_parallel_workers=1)
    test_dataset = ds.GeneratorDataset(test_data, column_names=['img', 'gt', 'label', 'idx'],
                                       shuffle=False, num_parallel_workers=1)

    train_dataset = train_dataset.batch(train_batch_size, drop_remainder=True, num_parallel_workers=1)
    test_dataset = test_dataset.batch(test_batch_size, drop_remainder=False, num_parallel_workers=1)

    return train_dataset, test_dataset
