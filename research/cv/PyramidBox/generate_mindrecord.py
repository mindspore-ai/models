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

import argparse
import os
from mindspore.mindrecord import FileWriter

from src.dataset import WIDERDataset
from src.config import cfg

parser = argparse.ArgumentParser(description='Generate Mindrecord File for training')
parser.add_argument('--prefix', type=str, default='./data', help="Directory to store mindrecord file")
parser.add_argument('--val_name', type=str, default='val.mindrecord', help='Name of val mindrecord file')

args = parser.parse_args()

def data_to_mindrecord(mindrecord_prefix, mindrecord_name, dataset):
    if not os.path.exists(mindrecord_prefix):
        os.mkdir(mindrecord_prefix)
    mindrecord_path = os.path.join(mindrecord_prefix, mindrecord_name)
    writer = FileWriter(mindrecord_path, 1, overwrite=True)

    data_json = {
        'img': {"type": "float32", "shape": [3, 640, 640]},
        'face_loc': {"type": "float32", "shape": [34125, 4]},
        'face_conf': {"type": "float32", "shape": [34125]},
        'head_loc': {"type": "float32", "shape": [34125, 4]},
        'head_conf': {"type": "float32", "shape": [34125]}
    }

    writer.add_schema(data_json, 'data_json')
    count = 0
    for d in dataset:
        img, face_loc, face_conf, head_loc, head_conf = d

        row = {
            "img": img,
            "face_loc": face_loc,
            "face_conf": face_conf,
            "head_loc": head_loc,
            "head_conf": head_conf
        }

        writer.write_raw_data([row])
        count += 1
    writer.commit()
    print("Total train data: ", count)
    print("Create mindrecord done!")


if __name__ == '__main__':
    print("Start generating val mindrecord file")
    ds_val = WIDERDataset(cfg.FACE.VAL_FILE, mode='val')
    data_to_mindrecord(args.prefix, args.val_name, ds_val)
