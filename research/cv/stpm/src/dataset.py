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
dataset
"""
import os
import glob
import numpy as np
from PIL import Image
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.transforms import Compose


class MVTecDataset():
    """MVTecDataset"""

    def __init__(self, root, transform, gt_transform, phase, is_json=False, save_sample=False):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.save_sample = save_sample
        self.is_json = is_json
        self.transform = transform
        self.gt_transform = gt_transform
        self.max_filename_length = 300
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        """load_dataset
        """
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

    def trans_filename(self, file_path, img_type):
        file_name = os.path.split(file_path)[-1]
        file_name = os.path.splitext(file_name)[0]
        file_name_type = img_type + "_" + file_name
        file_name_type = np.fromstring(file_name_type, np.uint8)
        file_name_type = np.pad(file_name_type, (0, self.max_filename_length - file_name_type.shape[0]))
        return file_name_type

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)[0]

        if gt == 0:
            gt = np.zeros((1, np.array(img).shape[-2], np.array(img).shape[-2])).tolist()
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)[0]

        if self.is_json:
            return os.path.basename(img_path[:-4]), img_type

        if self.save_sample:
            file_name = self.trans_filename(img_path, img_type)
            return img, gt, label, idx, file_name

        return img, gt, label, idx


def createDataset(dataset_path, category, save_sample=False, out_size=256, train_batch_size=32):
    """createDataset
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = Compose([
        vision.Resize((out_size, out_size), interpolation=Inter.ANTIALIAS),
        vision.CenterCrop(out_size),
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std, is_hwc=False)
    ])
    gt_transforms = Compose([
        vision.Resize((out_size, out_size)),
        vision.CenterCrop(out_size),
        vision.ToTensor()
    ])

    train_data = MVTecDataset(root=os.path.join(dataset_path, category),
                              transform=data_transforms, gt_transform=gt_transforms, phase='train')
    test_data = MVTecDataset(root=os.path.join(dataset_path, category),
                             transform=data_transforms, gt_transform=gt_transforms,
                             phase='test', save_sample=save_sample)

    train_dataset = ds.GeneratorDataset(train_data, column_names=['img', 'gt', 'label', 'idx'],
                                        shuffle=True)
    eval_column_names = ['img', 'gt', 'label', 'idx']
    if save_sample:
        eval_column_names.append('filename')
    test_dataset = ds.GeneratorDataset(test_data, column_names=eval_column_names,
                                       shuffle=False)

    train_dataset = train_dataset.batch(train_batch_size, drop_remainder=False)
    test_dataset = test_dataset.batch(1, drop_remainder=True)

    return train_dataset, test_dataset
