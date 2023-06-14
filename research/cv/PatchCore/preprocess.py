# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C2
import mindspore.dataset.vision as vision
from mindspore.common import set_seed
from mindspore.dataset.transforms.transforms import Compose
from mindspore.dataset.vision import Inter

from src.config import cfg
from src.dataset import MVTecDataset

set_seed(1)

parser = argparse.ArgumentParser(description="preprocesss")

parser.add_argument("--data_dir", type=str, default="")
parser.add_argument("--img_dir", type=str, help="")
parser.add_argument("--category", type=str, default="")

args = parser.parse_args()


def createDatasetJson(dataset_path, category, data_transforms, gt_transforms):
    """createDatasetJson"""
    path = os.path.join(dataset_path, "json", category)
    if not os.path.exists(path):
        os.makedirs(path)
    train_json_path = os.path.join(path, "{}_{}.json".format(category, "pre"))
    test_json_path = os.path.join(path, "{}_{}.json".format(category, "infer"))

    if not os.path.isfile(train_json_path):
        os.mknod(train_json_path)
        train_data = MVTecDataset(
            root=os.path.join(dataset_path, category),
            transform=data_transforms,
            gt_transform=gt_transforms,
            phase="train",
            is_json=True,
        )
        train_label = {}
        train_data_length = train_data.__len__()
        for i in range(train_data_length):
            single_label = {}
            name, img_type = train_data.__getitem__(i)
            single_label["name"] = name
            single_label["img_type"] = img_type
            train_label["{}".format(i)] = single_label

        json_path = Path(train_json_path)
        with json_path.open("w") as json_file:
            json.dump(train_label, json_file)

    if not os.path.isfile(test_json_path):
        os.mknod(test_json_path)
        test_data = MVTecDataset(
            root=os.path.join(dataset_path, category),
            transform=data_transforms,
            gt_transform=gt_transforms,
            phase="test",
            is_json=True,
        )
        test_label = {}
        test_data_length = test_data.__len__()
        for i in range(test_data_length):
            single_label = {}
            name, img_type = test_data.__getitem__(i)
            single_label["name"] = name
            single_label["img_type"] = img_type
            test_label["{}".format(i)] = single_label

        json_path = Path(test_json_path)
        with json_path.open("w") as json_file:
            json.dump(test_label, json_file)

    return train_json_path, test_json_path


def createDataset(dataset_path, category):
    """createDataset"""
    mean = [0.485, 0.456, 0.406]  # mean should be positive
    std = cfg.std_dft

    data_transforms = Compose(
        [
            vision.Resize((256, 256), interpolation=Inter.ANTIALIAS),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ]
    )
    gt_transforms = Compose([vision.Resize((256, 256)), vision.CenterCrop(224), vision.ToTensor()])

    train_json_path, test_json_path = createDatasetJson(dataset_path, category, data_transforms, gt_transforms)

    train_data = MVTecDataset(
        root=os.path.join(dataset_path, category), transform=data_transforms, gt_transform=gt_transforms, phase="train"
    )
    test_data = MVTecDataset(
        root=os.path.join(dataset_path, category), transform=data_transforms, gt_transform=gt_transforms, phase="test"
    )

    train_dataset = ds.GeneratorDataset(train_data, column_names=["img", "gt", "label", "idx"], shuffle=False)
    test_dataset = ds.GeneratorDataset(test_data, column_names=["img", "gt", "label", "idx"], shuffle=False)

    type_cast_float32_op = C2.TypeCast(mstype.float32)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="img")
    test_dataset = test_dataset.map(operations=type_cast_float32_op, input_columns="img")

    train_dataset = train_dataset.batch(1, drop_remainder=False)
    test_dataset = test_dataset.batch(1, drop_remainder=False)

    return train_dataset, test_dataset, train_json_path, test_json_path


if __name__ == "__main__":
    set_seed(1)
    train_dataset_, test_dataset_, train_json_path_, test_json_path_ = createDataset(args.data_dir, args.category)
    root_path = os.path.join(args.img_dir, args.category)
    train_path = os.path.join(root_path, "pre")
    test_path = os.path.join(root_path, "infer")
    label_path = os.path.join(root_path, "label")

    train_label_ = {}
    for j, data in enumerate(train_dataset_.create_dict_iterator()):
        train_single_lable = {}

        img = data["img"].asnumpy()
        gt = data["gt"].asnumpy()
        label = data["label"].asnumpy()
        idx = data["idx"].asnumpy()

        # save img
        file_name_img = "data_img" + "_" + str(j) + ".bin"
        file_path = os.path.join(train_path, file_name_img)
        img.tofile(file_path)

        train_single_lable["gt"] = gt.tolist()
        train_single_lable["label"] = label.tolist()
        train_single_lable["idx"] = idx.tolist()

        train_label_["{}".format(j)] = train_single_lable

    test_label_ = {}
    for j, data in enumerate(test_dataset_.create_dict_iterator()):
        test_single_lable = {}

        img = data["img"].asnumpy()
        gt = data["gt"].asnumpy()
        label = data["label"].asnumpy()
        idx = data["idx"].asnumpy()

        # save img
        file_name_img = "data_img" + "_" + str(j) + ".bin"
        file_path = os.path.join(test_path, file_name_img)
        img.tofile(file_path)

        test_single_lable["gt"] = gt.tolist()
        test_single_lable["label"] = label.tolist()
        test_single_lable["idx"] = idx.tolist()

        test_label_["{}".format(j)] = test_single_lable

    train_label_["pre_json_path"] = train_json_path_
    test_label_["infer_json_path"] = test_json_path_

    train_label_json_path = Path(os.path.join(label_path, "pre_label.json"))
    with train_label_json_path.open("w") as json_path_:
        json.dump(train_label_, json_path_)

    test_label_json_path = Path(os.path.join(label_path, "infer_label.json"))
    with test_label_json_path.open("w") as json_path_:
        json.dump(test_label_, json_path_)
