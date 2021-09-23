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
"""
generate annotation file for dataset ucf101.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def get_n_frames(video_path):
    """
    Get frame number for video.
    """
    return len([
        x for x in video_path.iterdir()
        if 'jpg' in x.name and x.name[0] != '.'
    ])


def convert_csv_to_dict(csv_path, subset):
    """
    Convert csv annotation file to dict.
    """
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        slash_rows = data.iloc[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1].split('.')[0]

        keys.append(basename)
        key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}

    return database


def load_labels(label_csv_path_):
    data = pd.read_csv(label_csv_path_, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.iloc[i, 1])
    return labels


def convert_ucf101_csv_to_json(label_csv_path_ucf101, train_csv_path_ucf101, val_csv_path_ucf101,
                               video_dir_path_ucf101, dst_json_path_ucf101):
    """
    Convert csv annotation file to dict for dataset ucf101.
    """
    labels = load_labels(label_csv_path_ucf101)
    train_database = convert_csv_to_dict(train_csv_path_ucf101, 'training')
    val_database = convert_csv_to_dict(val_csv_path_ucf101, 'validation')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path_ucf101 / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path_ucf101.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path',
                        default=None,
                        type=Path,
                        help=('Directory path including classInd.txt, '
                              'trainlist0-.txt, testlist0-.txt'))
    parser.add_argument('--video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('--dst_path',
                        default=None,
                        type=Path,
                        help='Directory path of dst json file.')

    args = parser.parse_args()

    for split_index in range(1, 4):
        label_csv_path = args.dir_path / 'classInd.txt'
        train_csv_path = args.dir_path / 'trainlist0{}.txt'.format(split_index)
        val_csv_path = args.dir_path / 'testlist0{}.txt'.format(split_index)
        dst_json_path = args.dst_path / 'ucf101_0{}.json'.format(split_index)

        convert_ucf101_csv_to_json(label_csv_path, train_csv_path, val_csv_path,
                                   args.video_path, dst_json_path)
