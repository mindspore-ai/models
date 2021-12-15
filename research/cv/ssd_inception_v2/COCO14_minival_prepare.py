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
"""Prepare minival json for COCO14 evaluation"""

import argparse
import json
import os
from shutil import copyfile

from tqdm import tqdm


def main():
    """Prepare the dataset"""
    parser = argparse.ArgumentParser(description='Generate minival_instances_val2014.json by instances_val2014.json')
    parser.add_argument('--val_annotation_json', type=str,
                        help='Path to instances_val2014.json')
    parser.add_argument('--minival_annotation_json', type=str,
                        help='Path to the output json')
    parser.add_argument('--mininval_idx', type=str,
                        help='Path to mscoco_minival_idx.txt')
    parser.add_argument('--val_images_folder', type=str,
                        help='Folder with validation images')
    parser.add_argument('--mininval_images_folder', type=str,
                        help='Folder to copy images only for minivalset')

    args = parser.parse_args()

    if not os.path.exists(args.mininval_images_folder):
        os.makedirs(args.mininval_images_folder)

    with open(args.val_annotation_json, 'r') as load_f:
        val_file = json.load(load_f)

    with open(args.mininval_idx, "r") as f:
        minival_ids = f.readlines()

    minival_images = []
    minival_annotations = []

    for minival_id in tqdm(minival_ids):
        m_id = minival_id.strip('\n')
        filename = 'COCO_val2014_' + m_id.zfill(12) + '.jpg'
        copyfile(
            os.path.join(args.val_images_folder, filename),
            os.path.join(args.mininval_images_folder, filename)
        )

    for minival_id in tqdm(minival_ids):
        m_id = int(minival_id.strip('\n'))
        for val_image in val_file['images']:
            if m_id == val_image['id']:
                minival_images.append(val_image)

    for minival_id in tqdm(minival_ids):
        m_id = int(minival_id.strip('\n'))
        for val_annote in val_file['annotations']:
            if m_id == val_annote['image_id']:
                minival_annotations.append(val_annote)

    print('minival_images', len(minival_images))
    print('minival_annotations', len(minival_annotations))
    print('minival_ids', len(minival_ids))

    minival_dict = {
        'images': minival_images,
        'annotations': minival_annotations,
        'categories': val_file['categories'],
    }

    target_path = '/'.join(args.minival_annotation_json.split('/')[:-1])
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    with open(args.minival_annotation_json, "w") as f:
        json.dump(minival_dict, f)


if __name__ == '__main__':
    main()
