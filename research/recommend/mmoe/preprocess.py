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
"""preprocess."""
import os

from src.load_dataset import create_dataset
from src.model_utils.config import config


def generate_bin():
    """generate bin files"""
    ds = create_dataset(config.data_path, batch_size=1,
                        training=False)
    batch_data_path = os.path.join(config.result_path, "00_batch_data")
    label1_path = os.path.join(config.result_path, "01_label1")
    label2_path = os.path.join(config.result_path, "02_label2")

    os.makedirs(batch_data_path)
    os.makedirs(label1_path)
    os.makedirs(label2_path)

    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
        file_name = "census_val_" + "_" + str(i) + ".bin"
        batch_data = data['data']
        batch_data.tofile(os.path.join(batch_data_path, file_name))

        label1 = data['income_labels']
        label1.tofile(os.path.join(label1_path, file_name))

        label2 = data['married_labels']
        label2.tofile(os.path.join(label2_path, file_name))

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    generate_bin()
