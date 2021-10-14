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
from src.dataset import create_dataset_cifar10
parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, default='', help='eval data dir')

args = parser.parse_args()
if __name__ == "__main__":
    dataset = create_dataset_cifar10(args.data_path, 1, 1, False)
    img_path = os.path.join('./preprocess_Result/', "img_data")
    label_path = os.path.join('./preprocess_Result/', "label")
    os.makedirs(img_path)
    os.makedirs(label_path)
    batch_size = 1
    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_label = data["label"]
        file_name = "vit_base_cifar10_" + str(batch_size) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)
        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
