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
"""preprocess data"""
import os

from src.dataset import create_dataset, DataType
from src.model_utils.config import config

def generate_bin():
    '''generate bin files for inference
    '''

    ds = create_dataset(config.dataset_path, train_mode=False,
                        epochs=1, batch_size=config.test_batch_size,
                        line_per_sample=1,
                        data_type=DataType(config.data_format))
    batch_cats_path = os.path.join(config.result_path, "00_batch_cats")
    batch_nums_path = os.path.join(config.result_path, "01_batch_nums")
    labels_path = os.path.join(config.result_path, "02_labels")

    os.makedirs(batch_cats_path, exist_ok=True)
    os.makedirs(batch_nums_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
        file_name = "criteo_bs" + str(config.test_batch_size) + "_" + str(i) + ".bin"
        batch_cats = data['feat_ids']
        batch_cats.tofile(os.path.join(batch_cats_path, file_name))

        batch_nums = data['feat_vals']
        batch_nums.tofile(os.path.join(batch_nums_path, file_name))

        labels = data['label']
        labels.tofile(os.path.join(labels_path, file_name))

    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    generate_bin()
