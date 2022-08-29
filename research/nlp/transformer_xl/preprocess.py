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
"""Transformer preprocess."""

import os

from mindspore.dataset import GeneratorDataset

from src.model_utils.config import config
from src.utils.dataset_util import get_dataset


def generate_bin():
    '''
    Generate bin files.
    '''
    datadir = config.datadir
    dataset = config.dataset

    dataset = get_dataset(datadir, dataset)
    test_dataset = GeneratorDataset(source=dataset.get_test_generator(),
                                    column_names=['data', 'target'], shuffle=False)
    cur_dir = config.result_path

    source_eos_ids_path = os.path.join(cur_dir, "00_data")
    source_eos_mask_path = os.path.join(cur_dir, "01_target")

    if not os.path.isdir(source_eos_ids_path):
        os.makedirs(source_eos_ids_path)
    if not os.path.isdir(source_eos_mask_path):
        os.makedirs(source_eos_mask_path)

    batch_size = config.batch_size

    for i, data in enumerate(test_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        if i < 10:
            sort_no = "0" + str(i)
        else:
            sort_no = str(i)
        file_name = "transformer_bs_" + str(batch_size) + "_" + sort_no + ".bin"
        source_eos_ids = data['data']
        source_eos_ids.tofile(os.path.join(source_eos_ids_path, file_name))

        source_eos_mask = data['target']
        source_eos_mask.tofile(os.path.join(source_eos_mask_path, file_name))

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    generate_bin()
