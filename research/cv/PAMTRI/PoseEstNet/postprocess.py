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
"""postprocess"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from src.dataset.veri import VeRiDataset
from src.config import cfg, update_config
from src.dataset import flip_back, get_final_preds, get_label

parser = argparse.ArgumentParser(description='postprocess')

parser.add_argument('--input_dir', type=str, default="")
parser.add_argument('--result_dir', type=str, default="")
parser.add_argument('--label_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--cfg', type=str, default='')

args = parser.parse_args()

def _print_name_value(_name_value, _full_arch_name):
    """print acc"""
    names = _name_value.keys()
    values = _name_value.values()
    num_values = len(_name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('| --- ' * (num_values+1) + '|')

    if len(_full_arch_name) > 15:
        _full_arch_name = _full_arch_name[:8] + '...'
    print(
        '| ' + _full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )

if __name__ == '__main__':
    update_config(cfg, args)

    data = VeRiDataset(cfg, args.data_dir, False)
    num_samples = len(data)
    all_preds = np.zeros(
        (num_samples, cfg.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    label_json_path = Path(args.label_dir)
    with label_json_path.open('r') as dst_file:
        all_labels = json.load(dst_file)

    json_path = get_label(cfg, args.data_dir)
    dst_json_path = Path(json_path)
    with dst_json_path.open('r') as dst_file:
        allImage = json.load(dst_file)

    for i in range(len(os.listdir(args.input_dir + "img/"))):
        path_one = args.result_dir + "veri_data_img_{}_0.bin".format(i)
        output = np.fromfile(path_one, dtype=np.float32).reshape(32, 36, 64, 64)
        path_two = args.result_dir + "veri_data_imgFlip_{}_0.bin".format(i)
        output_flipped = np.fromfile(path_two, dtype=np.float32).reshape(32, 36, 64, 64)

        single_label = all_labels["{}".format(i)]
        center = np.array(single_label['center'])
        scale = np.array(single_label['scale'])
        score = np.array(single_label['score'])
        image_label = single_label['image']

        image = []
        for j in range(32):
            image.append(allImage['{}'.format(image_label[j])])

        output_flipped = flip_back(output_flipped, data.flip_pairs)
        output_flipped_copy = output_flipped
        output_flipped[:, :, :, 1:] = output_flipped_copy[:, :, :, 0:-1]
        output = (output + output_flipped) * 0.5

        num_images = 32
        preds, maxvals = get_final_preds(cfg, output, center, scale)
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(image)
        idx += num_images

        output_dir = ""
        name_values, perf_indicator = data.evaluate(
            all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    print("310 acc is {}".format(perf_indicator))
