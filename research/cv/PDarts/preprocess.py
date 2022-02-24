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
import os
import argparse
import numpy as np
from src.dataset import create_cifar10_dataset

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, default='../cifar-10-binary/val', help='eval data dir')
result_path = './preprocess_Result/'
batch_size = 1
args = parser.parse_args()
if __name__ == "__main__":
    dataset = create_cifar10_dataset(args.data_path, training=False, num_parallel_workers=2, batch_size=batch_size)
    img_path = os.path.join(result_path, "00_data")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    label_list = []
    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "pdarts_data_bs" + str(batch_size) + "_" + str(idx) + ".bin"
        file_path = os.path.join(img_path, file_name)
        data["image"].tofile(file_path)
        label_list.append(data["label"])
    np.save(os.path.join(result_path, "cifar10_label_ids.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)
        