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
"""Convert Kaldi dataset"""

import argparse
import json
import pickle
from copy import deepcopy
from pathlib import Path

import kaldi_io


def convert_to_pickle(data_json_path, new_root_dir):
    """Convert kaldi dataset files"""

    with Path(data_json_path).open('r', encoding="utf-8") as file:
        dataset_dict = json.load(file)

    new_dataset_dict = dict()
    for sample_name, sample_info in dataset_dict['utts'].items():
        new_sample_info = deepcopy(sample_info)
        feature_path = sample_info['input'][0]['feat']
        feature = kaldi_io.read_mat(feature_path)

        new_feature_path = Path(new_root_dir) / (Path(feature_path).name.replace(':', '_') + '.pickle')
        with new_feature_path.open('wb') as file:
            pickle.dump(feature, file)
        new_sample_info['input'][0]['feat'] = new_feature_path.as_posix()
        new_dataset_dict[sample_name] = new_sample_info

    with (Path(new_root_dir) / 'data.json').open('w') as file:
        json.dump(new_dataset_dict, file, indent=2)


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dataset-path')
    args = parser.parse_args()
    for dataset_split in ['train', 'dev', 'test']:
        json_path = Path(args.processed_dataset_path) / 'dump' / dataset_split / 'deltafalse/data.json'
        new_root_dir = Path(args.processed_dataset_path) / 'pickled_dataset' / dataset_split
        new_root_dir.mkdir(exist_ok=True, parents=True)
        convert_to_pickle(json_path, new_root_dir)


if __name__ == '__main__':
    main()
