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
"""preprocess"""
import json
import argparse
from pathlib import Path
import numpy as np
from src.dataset.dataset import create_dataset
from src.config import cfg, update_config

parser = argparse.ArgumentParser(description='preprocess')

parser.add_argument("--result_dir", type=str, help="")
parser.add_argument("--label_dir", type=str, help="")
parser.add_argument('--cfg', type=str, default="")
parser.add_argument('--data_dir', type=str, default="")

args = parser.parse_args()

if __name__ == '__main__':
    update_config(cfg, args)

    data, dataset = create_dataset(cfg, args.data_dir, is_train=False)

    result_path = args.result_dir + "img/"
    result_path_flip = args.result_dir + "img_flip/"
    label_path = args.label_dir

    label_list = {}
    for i, data in enumerate(dataset.create_dict_iterator()):
        single_label_list = {}

        _input = data["input"].asnumpy()
        input_flip = np.flip(_input, 3)

        center = data["center"].asnumpy()
        scale = data["scale"].asnumpy()
        score = data["score"].asnumpy()
        image_label = data["image"].asnumpy()
        joints = data["joints"].asnumpy()
        joints_vis = data["joints_vis"].asnumpy()

        file_name = "veri_data_img" + "_" + str(i) + ".bin"
        file_path = result_path + file_name
        _input.tofile(file_path)
        file_name_flip = "veri_data_imgFlip" + "_" + str(i) + ".bin"
        file_path_flip = result_path_flip + file_name_flip
        input_flip.tofile(file_path_flip)

        single_label_list['center'] = center.tolist()
        single_label_list['scale'] = scale.tolist()
        single_label_list['score'] = score.tolist()
        single_label_list['image'] = image_label.tolist()

        label_list['{}'.format(i)] = single_label_list

    json_path = Path(label_path + 'label.json')
    with json_path.open('w') as dst_json:
        json.dump(label_list, dst_json)
