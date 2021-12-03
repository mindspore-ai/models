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
Use this file to generate ascend310 inference datasets
"""
import pickle
import argparse
from src.dataset import create_dataset_val


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='get ecolite train dataset')
    parser.add_argument('dataset', type=str, default="ucf101",
                        choices=['ucf101', 'hmdb51', 'kinetics', 'something', 'jhmdb'])
    parser.add_argument('modality', type=str, default="RGB", choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('val_list', type=str)
    parser.add_argument('valdataset_path', type=str)
    parser.add_argument('--arch', type=str, default="ECO")
    parser.add_argument('--num_segments', type=int, default=4)
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--flow_prefix', default="", type=str)
    parser.add_argument('--rgb_prefix', default="img_", type=str)
    return parser.parse_args()


args = get_args()


def run():
    """get 310 data set"""
    if args.dataset == 'ucf101':
        rgb_read_format = "{:06d}.jpg"
    elif args.dataset == 'hmdb51':
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'kinetics':
        rgb_read_format = "{:04d}.jpg"
    elif args.dataset == 'something':
        rgb_read_format = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    val_dataset = create_dataset_val(args, rgb_read_format)
    for i, data in enumerate(val_dataset.create_dict_iterator()):
        data['image'].asnumpy().tofile(args.valdataset_path + '/eval_data_' + str(i) + '_.bin')
        output_data = open('label/eval_label_' + str(i) + '.pkl', 'wb')
        pickle.dump(data['label'].asnumpy(), output_data)
        output_data.close()


if __name__ == '__main__':
    run()
