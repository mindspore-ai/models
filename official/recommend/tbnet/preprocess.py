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
"""TB-Net evaluation."""

import os
import argparse
import shutil
import math
import numpy as np

from mindspore import context

from src import config, dataset


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Preprocess TBNet training data.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='test.csv',
        help="the csv datafile inside the dataset folder (e.g. test.csv)"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--device_target',
        type=str,
        required=False,
        default='Ascend',
        choices=['Ascend', 'GPU'],
        help="run code on GPU or Ascend NPU"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def preprocess_tbnet():
    """Data preprocess for inference."""
    args = get_args()

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    test_csv_path = os.path.join(home, 'data', args.dataset, args.csv)

    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    print(f"creating dataset from {test_csv_path}...")
    net_config = config.TBNetConfig(config_path)
    if args.device_target == 'Ascend':
        net_config.per_item_paths = math.ceil(net_config.per_item_paths / 16) * 16
    eval_ds = dataset.create(test_csv_path, net_config.per_item_paths, train=True).batch(1)
    item_path = os.path.join('./preprocess_Result/', '00_item')
    rl1_path = os.path.join('./preprocess_Result/', '01_rl1')
    ety_path = os.path.join('./preprocess_Result/', '02_ety')
    rl2_path = os.path.join('./preprocess_Result/', '03_rl2')
    his_path = os.path.join('./preprocess_Result/', '04_his')
    rate_path = os.path.join('./preprocess_Result/', '05_rate')
    rst_path = [item_path, rl1_path, ety_path, rl2_path, his_path, rate_path]
    if os.path.isdir('./preprocess_Result/'):
        shutil.rmtree('./preprocess_Result/')
    for paths in rst_path:
        os.makedirs(paths)

    idx = 0
    for d in eval_ds.create_dict_iterator():
        item_rst = d['item'].asnumpy().astype(np.int)
        rl1_rst = np.expand_dims(d['relation1'].asnumpy().astype(np.int), axis=0)
        ety_rst = np.expand_dims(d['entity'].asnumpy().astype(np.int), axis=0)
        rl2_rst = np.expand_dims(d['relation2'].asnumpy().astype(np.int), axis=0)
        his_rst = np.expand_dims(d['hist_item'].asnumpy().astype(np.int), axis=0)
        rate_rst = d['rating'].asnumpy().astype(np.float32)

        item_name = 'tbnet_item_bs1_' + str(idx) + '.bin'
        rl1_name = 'tbnet_rl1_bs1_' + str(idx) + '.bin'
        ety_name = 'tbnet_ety_bs1_' + str(idx) + '.bin'
        rl2_name = 'tbnet_rl2_bs1_' + str(idx) + '.bin'
        his_name = 'tbnet_his_bs1_' + str(idx) + '.bin'
        rate_name = 'tbnet_rate_bs1_' + str(idx) + '.bin'

        item_real_path = os.path.join(item_path, item_name)
        rl1_real_path = os.path.join(rl1_path, rl1_name)
        ety_real_path = os.path.join(ety_path, ety_name)
        rl2_real_path = os.path.join(rl2_path, rl2_name)
        his_real_path = os.path.join(his_path, his_name)
        rate_real_path = os.path.join(rate_path, rate_name)

        item_rst.tofile(item_real_path)
        rl1_rst.tofile(rl1_real_path)
        ety_rst.tofile(ety_real_path)
        rl2_rst.tofile(rl2_real_path)
        his_rst.tofile(his_real_path)
        rate_rst.tofile(rate_real_path)

        idx += 1


if __name__ == '__main__':
    preprocess_tbnet()
