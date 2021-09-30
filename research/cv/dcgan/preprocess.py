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
"""preprocess"""
import os
import argparse
import numpy as np
from src.config import dcgan_cifar10_cfg
from src.dataset import create_dataset_cifar10

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--dataset_name', type=str, default="cifar10")
parser.add_argument('--data_path', type=str, default='', help='eval data dir')

args = parser.parse_args()
if __name__ == "__main__":
    dataset_train = create_dataset_cifar10(args.data_path, num_parallel_workers=2, usage='train')
    img_path_train = os.path.join('./preprocess_Result/', "train_data")
    os.makedirs(img_path_train)
    label_list = []
    for idx, data in enumerate(dataset_train.create_dict_iterator(output_numpy=True)):
        file_name = "dcgan_data_bs" + str(dcgan_cifar10_cfg.batch_size) + "_" + str(idx) + ".bin"
        file_path = os.path.join(img_path_train, file_name)
        data["image"].tofile(file_path)
        label_list.append(data["label"])
    np.save(os.path.join('./preprocess_Result/', "cifar10_label_ids_train.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)

    dataset_test = create_dataset_cifar10(args.data_path, num_parallel_workers=2, usage='test')
    img_path_test = os.path.join('./preprocess_Result/', "test_data")
    os.makedirs(img_path_test)
    label_list = []
    for idx, data in enumerate(dataset_test.create_dict_iterator(output_numpy=True)):
        file_name = "dcgan_data_bs" + str(dcgan_cifar10_cfg.batch_size) + "_" + str(idx) + ".bin"
        file_path = os.path.join(img_path_test, file_name)
        data["image"].tofile(file_path)
        label_list.append(data["label"])
    np.save(os.path.join('./preprocess_Result/', "cifar10_label_ids_test.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)
