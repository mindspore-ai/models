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
"""Preprocess api."""
import os

from src.dataset import load_dataset
from src.model_utils.config import config


def generate_txt():
    '''
    Generate bin files.
    '''
    config.batch_size = 1
    config.result_path = "./infer_data"
    ds = load_dataset(data_files=config.test_dataset,
                      batch_size=config.batch_size,
                      epoch_count=1,
                      sink_mode=config.dataset_sink_mode,
                      shuffle=False) if config.test_dataset else None
    cur_dir = config.result_path
    source_eos_ids_path = os.path.join(cur_dir, "source_ids.txt")
    source_eos_mask_path = os.path.join(cur_dir, "mask_ids.txt")
    target_eos_ids_path = os.path.join(cur_dir, "target_ids.txt")

    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)

    source_eos_ids_txt = ""
    source_eos_mask_txt = ""
    target_eos_ids_txt = ""
    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        print("batch %i:"%i)
        source_eos_ids = data['source_eos_ids'][0]
        for j in range(len(source_eos_ids)):
            source_eos_ids_txt = source_eos_ids_txt + "%i " % (source_eos_ids[j])
        source_eos_ids_txt += '\n'

        source_eos_mask = data['source_eos_mask'][0]
        for j in range(len(source_eos_mask)):
            source_eos_mask_txt = source_eos_mask_txt + "%i " % (source_eos_mask[j])
        source_eos_mask_txt += '\n'

        target_eos_ids = data['target_eos_ids'][0]
        for j in range(len(target_eos_ids)):
            target_eos_ids_txt = target_eos_ids_txt + "%i " % (target_eos_ids[j])
        target_eos_ids_txt += '\n'

    with open(source_eos_ids_path, "w") as f:
        f.write(source_eos_ids_txt)
    with open(source_eos_mask_path, "w") as f:
        f.write(source_eos_mask_txt)
    with open(target_eos_ids_path, "w") as f:
        f.write(target_eos_ids_txt)
    print("=" * 20, "export txt files finished", "=" * 20)


if __name__ == '__main__':
    generate_txt()
