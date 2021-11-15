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
"""Evaluation for DeepID"""
import time
import argparse

import mindspore
from mindspore import context, load_checkpoint
from mindspore.train.serialization import load_param_into_net
import mindspore.ops as P

from src.utils import get_network
from src.dataset import dataloader

parser = argparse.ArgumentParser(description='DeepID_test')

parser.add_argument('--data_url', type=str, default='data/', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')
parser.add_argument('--ckpt_url', type=str, default='', help='checkpoint path')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--device_target", type=str, default='Ascend', help="device id, default: 0.")
parser.add_argument('--num_class', type=int, default=1283, help='num_class')
parser.add_argument('--input_dim', type=int, default=3, help='input dim')
parser.add_argument('--mode', type=str, default='valid', help='dataset mode')

if __name__ == '__main__':
    args_opt = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)

    valid_dataset, valid_dataset_length = dataloader(args_opt.data_url, epoch=1,
                                                     mode=args_opt.mode, batch_size=args_opt.batch_size)

    valid_dataset_iter = valid_dataset.create_dict_iterator()
    print('Valid dataset length:', valid_dataset_length)

    deepid = get_network(args_opt, args_opt.num_class)
    param_network = load_checkpoint(args_opt.ckpt_url)
    load_param_into_net(deepid, param_network)

    print('Start Testing!')

    correct_num = 0
    for data in valid_dataset_iter:
        step_begin_time = time.time()
        img_valid = data['image']
        label = data['label']
        pred_y = P.Argmax(axis=1)(deepid(img_valid))
        result = P.cast(P.Equal()(pred_y, label), mindspore.float16)
        correct_num += P.reduce_sum(result)
        print('Test time per step: {:.2f} ms'.format((time.time()-step_begin_time)*1000))
    print('Valid dataset accuracy: {:.2f}%'.format(100*correct_num.asnumpy()/valid_dataset_length))
