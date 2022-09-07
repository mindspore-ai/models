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
"""310 data_processing"""
import os
from src.dataset import create_eval_dataset
from src.config import infer_config, symbols
from src.decoder import GreedyDecoder
import numpy as np


def preprocess_data():
    config = infer_config

    ds = create_eval_dataset(data_dir=config.DataConfig.Data_dir,
                             manifest_filepath=config.DataConfig.test_manifest,
                             labels=symbols,
                             batch_size=config.batch_size_infer,
                             train_mode=False)

    target_decoder = GreedyDecoder(symbols, blank_index=len(symbols)-1)

    feature_path = os.path.join(config.result_path, "00_data")
    length_path = os.path.join(config.result_path, "01_data")
    os.makedirs(feature_path)
    os.makedirs(length_path)

    with open('target.txt', 'w', encoding='utf-8') as f:
        for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
            file_name = "jasper_bs_" + str(
                config.batch_size_infer) + "_" + str(i) + ".bin"
            data['inputs'].tofile(os.path.join(feature_path, file_name))
            data['input_length'].tofile(os.path.join(length_path, file_name))

            target_indices, targets = data['target_indices'], data['targets']
            split_targets = []
            start, count, last_id = 0, 0, 0
            for j in range(np.shape(targets)[0]):
                if target_indices[j, 0] == last_id:
                    count += 1
                else:
                    split_targets.append(list(targets[start:count]))
                    last_id += 1
                    start = count
                    count += 1
            split_targets.append(list(targets[start:]))
            target_strings = target_decoder.convert_to_strings(split_targets)
            f.write(' '.join(target_strings[0]) + '\n')

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    preprocess_data()
