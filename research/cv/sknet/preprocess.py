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
from src.dataset import create_dataset_cifar10
parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, default='', help='eval data dir')

args = parser.parse_args()
if __name__ == "__main__":
    dataset = create_dataset_cifar10(args.data_path, False)
    img_path = os.path.join('./preprocess_Result/', "00_data")
    os.makedirs(img_path)
    label_list = []
    batch_size = 32
    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "sknet_data_bs" + str(batch_size) + "_" + str(idx) + ".bin"
        file_path = os.path.join(img_path, file_name)
        data["image"].tofile(file_path)
        label_list.append(data["label"])
    np.save(os.path.join('./preprocess_Result/', "cifar10_label_ids.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)
