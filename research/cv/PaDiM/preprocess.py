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
"""preprocess"""
import argparse
import json
import os
from pathlib import Path
from collections import namedtuple
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision import Inter

from src.dataset import MVTecDataset

parser = argparse.ArgumentParser(description='preprocesss')

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument("--img_dir", type=str, help="")
parser.add_argument('--category', type=str, default='')

args = parser.parse_args()

def createDataset(dataset_path, category):
    """createDataset"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = Compose([
        py_vision.Resize((256, 256), interpolation=Inter.ANTIALIAS),
        py_vision.CenterCrop(224),
        py_vision.ToTensor(),
        py_vision.Normalize(mean=mean, std=std)
    ])
    gt_transforms = Compose([
        py_vision.Resize((256, 256), interpolation=Inter.NEAREST),
        py_vision.CenterCrop(224),
        py_vision.ToTensor()
    ])

    train_data = MVTecDataset(root=os.path.join(dataset_path, category),
                              transform=data_transforms, gt_transform=gt_transforms, phase='train')
    test_data = MVTecDataset(root=os.path.join(dataset_path, category),
                             transform=data_transforms, gt_transform=gt_transforms, phase='test')
    train_dataset_len = len(train_data.img_paths)
    test_dataset_len = len(test_data.img_paths)
    train_dataset = ds.GeneratorDataset(train_data, column_names=['img', 'gt', 'label', 'idx'],
                                        shuffle=False)
    test_dataset = ds.GeneratorDataset(test_data, column_names=['img', 'gt', 'label', 'idx'],
                                       shuffle=False)

    type_cast_float32_op = C2.TypeCast(mstype.float32)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="img")
    test_dataset = test_dataset.map(operations=type_cast_float32_op, input_columns="img")

    train_dataset = train_dataset.batch(1, drop_remainder=False)
    test_dataset = test_dataset.batch(1, drop_remainder=False)

    res = namedtuple("res", ["train_dataset", "train_dataset_len", "test_dataset", "test_dataset_len"])
    return res(train_dataset, train_dataset_len, test_dataset, test_dataset_len)

if __name__ == '__main__':
    train_dataset_, _, test_dataset_, _ = createDataset(args.data_dir, args.category)
    root_path = os.path.join(args.img_dir, args.category)
    train_path = os.path.join(root_path, 'pre')
    test_path = os.path.join(root_path, 'infer')
    label_path = os.path.join(root_path, 'label')
    for j, data in enumerate(train_dataset_.create_dict_iterator()):
        img = data['img'].asnumpy()
        # save img
        file_name_img = "data_img" + "_" + str(j) + ".bin"
        file_path = os.path.join(train_path, file_name_img)
        img.tofile(file_path)
    test_label_ = {}
    for j, data in enumerate(test_dataset_.create_dict_iterator()):
        test_single_lable = {}
        img = data['img'].asnumpy()
        gt_mask_list = data['gt'].asnumpy()
        gt_list = data['label'].asnumpy()
        # save img
        file_name_img = "data_img" + "_" + str(j) + ".bin"
        file_path = os.path.join(test_path, file_name_img)
        img.tofile(file_path)

        test_single_lable['gt'] = gt_mask_list.tolist()
        test_single_lable['label'] = gt_list.tolist()

        test_label_['{}'.format(j)] = test_single_lable

    test_label_json_path = Path(os.path.join(label_path, 'infer_label.json'))
    with test_label_json_path.open('w') as json_path_:
        json.dump(test_label_, json_path_)
