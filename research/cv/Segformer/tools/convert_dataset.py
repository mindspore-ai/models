# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse
import os.path as osp

import mmcv
from cityscapesscripts.preparation.json2labelImg import json2labelImg


def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, "gtFine")

    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    if args.nproc > 1:
        mmcv.track_parallel_progress(convert_json_to_label, poly_files, args.nproc)
    else:
        mmcv.track_progress(convert_json_to_label, poly_files)


if __name__ == '__main__':
    main()
