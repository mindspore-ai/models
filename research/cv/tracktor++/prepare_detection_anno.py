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

"""Preprocessing for annotations"""

import argparse
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict


TRAIN_SEQUENCES = [
    'MOT17-02',
    'MOT17-04',
    'MOT17-05',
    'MOT17-10',
    'MOT17-11',
    'MOT17-13',
]


def extract_annotation(raw_anno):
    """Extract annotations from raw text line"""
    annos = raw_anno.split(',')
    if int(annos[6]) == 1 and int(annos[7]) == 1 and float(annos[8]) >= 0.25:
        img_id = int(annos[0])
        x_min = int(annos[2])
        y_min = int(annos[3])
        x_max = x_min + int(annos[4])
        y_max = y_min + int(annos[5])
        return img_id, [x_min, y_min, x_max, y_max]

    return None


def process_sequence(dataset_root_dir: str, sequence_name: str) -> Dict[str, list]:
    """Process annotations for one sequence"""
    mot_training_dataset = defaultdict(list)

    with (Path(dataset_root_dir) / f'train/{sequence_name}/gt/gt.txt').open('r') as file:
        raw_tracking_gt = file.read().split('\n')[:-1]

    for raw_anno in raw_tracking_gt:
        proc_anno = extract_annotation(raw_anno)
        if proc_anno is None:
            continue

        img_id, bbox = proc_anno
        mot_training_dataset[f'{sequence_name}/img1/{img_id:06d}.jpg'].append(bbox)

    return mot_training_dataset


def convert_to_text_lines(mot_training_dataset):
    """Convert annotations to text lines"""
    txt_lines = []
    for img_path, bboxes in mot_training_dataset.items():
        img_line = img_path + ' '
        for box in bboxes:
            img_line = img_line + f'{box[0]},{box[1]},{box[2]},{box[3]},1 '
        txt_lines.append(img_line + '\n')
    return txt_lines


def main():
    """Preprocess annotations"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='./MOT17DET/', type=str, help="Path to MOT17DET dataset")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    mot_training_dataset = defaultdict(list)
    for sequence_name in TRAIN_SEQUENCES:
        mot_training_dataset.update(process_sequence(dataset_path, sequence_name))

    txt_lines = convert_to_text_lines(mot_training_dataset)
    random.shuffle(txt_lines)
    with (Path(dataset_path) / 'train/shuffled_det_annotations.txt').open('w') as file:
        file.writelines(txt_lines)


if __name__ == '__main__':
    main()
