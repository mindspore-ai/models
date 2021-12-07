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
preprocess dataset for infer on ascend 310.
"""
import os
import argparse
import numpy as np

from src.dataset import create_dataset
from src.transforms import Stack, ToTorchFormatTensor, GroupNormalize, GroupScale, GroupCenterCrop, GroupOverSample

parser = argparse.ArgumentParser('mindspore tsn testing')
parser.add_argument('--dataset', type=str, default="ucf101", choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--modality', type=str, default="Flow", choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--test_list', type=str, default="")
parser.add_argument('--dataset_path', type=str, default="")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()

if __name__ == "__main__":

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
        args.flow_prefix = "flow_"

    crop_size = 224
    scale_size = 224 * 256 // 224
    input_mean = [104, 117, 128]
    input_std = [1]

    if args.modality == 'Flow':
        input_mean = [128]
    elif args.modality == 'RGBDiff':
        input_mean = input_mean * (1 + data_length)

    transform = []

    if args.test_crops == 1:
        transform.append(GroupScale(scale_size))
        transform.append(GroupCenterCrop(crop_size))
    elif args.test_crops == 10:
        transform.append(GroupOverSample(crop_size, scale_size))

    transform.append(Stack(roll=args.arch == 'BNInception'))
    transform.append(ToTorchFormatTensor(div=args.arch != 'BNInception'))
    transform.append(GroupNormalize(input_mean, input_std))

    image_tmpl = "img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg"
    data_loader = create_dataset(root_path=args.dataset_path, list_file=args.test_list,\
            batch_size=1, num_segments=args.test_segments, new_length=data_length,\
                modality=args.modality, image_tmpl=image_tmpl, transform=transform, test_mode=2, run_distribute=False)
    img_path = os.path.join(args.result_path, "00_data")

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    label_list = []
    # dataset is an instance of Dataset object
    iterator = data_loader.create_dict_iterator(num_epochs=1, output_numpy=True)
    for i, data in enumerate(iterator):
        file_name = "TSN_data_bs" + str(1) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        item = data['input']
        item = item.reshape((-1, length, item.shape[2], item.shape[3]))
        if args.modality == 'RGBDiff':
            reverse = list(range(data_length, 0, -1))
            input_c = 3
            input_view = item.reshape((-1, args.test_segments, data_length + 1, input_c,) + item.shape[2:])

            new_data = input_view[:, :, 1:, :, :, :].copy()
            for x in reverse:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            item = new_data

        item.tofile(file_path)
        label_list.append(data['label'])

    np.save(args.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
