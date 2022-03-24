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
"""Pre-process the dataset to get the mark-up lists"""

import argparse
from pathlib import Path

from tqdm import tqdm

_LABELS_DEFAULT_NAME = 'labels.csv'
_INPUT_VALIDATION_NAME = 'validation.csv'
_INPUT_TRAIN_FILE = 'train.csv'

_INPUT_DIR_NAME = '20bn-jester-v1'

_OUT_CATEGORIES_FILE = 'categories.txt'
_OUT_VALIDATION_FILE = 'val_videofolder.txt'
_OUT_TRAIN_FILE = 'train_videofolder.txt'


def _read_labels(labels_path: Path):
    with labels_path.open('r') as file:
        categories = [line.strip() for line in file]

    return sorted(categories)


def _save_categories(categories: list, output_path: Path):
    with output_path.open('w') as file:
        file.write('\n'.join(categories))


def _create_files_list(
        data_dir_path: Path,
        input_file_path: Path,
        output_file_path: Path,
        categories_map: dict,
):
    with input_file_path.open('r') as input_file:
        file_lines = input_file.readlines()

    output_data = []
    bad_files = 0

    for input_line in tqdm(file_lines):
        sub_dir_name, category_name = input_line.strip().split(';')
        category_index = categories_map[category_name]

        sub_dir_path = data_dir_path / sub_dir_name
        if not sub_dir_path.exists():
            bad_files += 1
            continue

        number_of_frames = len(list(sub_dir_path.glob('*.jpg')))
        output_data.append(f'{sub_dir_name} {number_of_frames} {category_index}')

    if bad_files:
        print(f'Unable to locate: {bad_files} files')

    with output_file_path.open('w') as output_file:
        output_file.write('\n'.join(output_data))


def preprocess_dataset():
    """
    Prepare labels jester dataset
    Use files labels.csv, validation.csv, train.csv from labels_path directory.
    Add categories.txt, train_videofolder.txt, val_videofolder.txt to save_labels_path directory
    """

    parser = argparse.ArgumentParser(description='process jester dataset labels.')
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('--labels_path', type=str, default="./labels")
    parser.add_argument('--save_labels_path', type=str, default=".")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.save_labels_path == ".":
        save_labels_path = dataset_root
    else:
        save_labels_path = Path(args.labels_path)

    if args.labels_path == "./labels":
        labels_path = dataset_root / "labels"
    else:
        labels_path = Path(args.save_labels_path)

    # Read categories list, sort it and save it in a new file
    categories = _read_labels(labels_path / _LABELS_DEFAULT_NAME)
    _save_categories(categories, dataset_root / _OUT_CATEGORIES_FILE)

    # Prepare mapping for categories (from names to indices)
    dict_categories = dict(zip(categories, range(len(categories))))

    # Save training files
    print("dataset path:", dataset_root)
    print("labels path:", labels_path)
    print("labels save to:", save_labels_path)
    print()

    print('Prepare training folders list')
    _create_files_list(
        data_dir_path=dataset_root / _INPUT_DIR_NAME,
        input_file_path=labels_path / _INPUT_TRAIN_FILE,
        output_file_path=save_labels_path / _OUT_TRAIN_FILE,
        categories_map=dict_categories,
    )
    # Save validation files
    print('Prepare validation folders list')
    _create_files_list(
        data_dir_path=dataset_root / _INPUT_DIR_NAME,
        input_file_path=labels_path / _INPUT_VALIDATION_NAME,
        output_file_path=save_labels_path / _OUT_VALIDATION_FILE,
        categories_map=dict_categories,
    )


if __name__ == '__main__':
    preprocess_dataset()
