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
"""
Eval the Autoslim resnet
"""
import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore import Model
from mindspore import load_checkpoint
from mindspore.common import set_seed

from src.dataset import data_transforms, create_dataset
from src.autoslim_resnet_for_val import AutoSlimModel

set_seed(1)

def main():
    """eval model"""
    parser = argparse.ArgumentParser(description='AutoSlim MindSpore Training')
    # run modelarts
    parser.add_argument('--data_url', type=str, default='', help='')
    parser.add_argument('--train_url', type=str, default='', help='')
    parser.add_argument('--ckpt_url', type=str, default='', help='')
    # device
    parser.add_argument('--run_modelarts', type=bool, default=False, help='')
    parser.add_argument('--distribute', type=bool, default=False, help='choice of distribute train')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='number of device which is chosen')
    parser.add_argument('--device_num', type=int, default=1, help='')
    parser.add_argument('--rank_id', type=int, default=0, help='')
    parser.add_argument('--random_seed', type=int, default=0, help='number of random seed which is chosen')
    # train parameters
    parser.add_argument('--dataset_path', type=str, default='/data/imagenet', help='The path of your imagenet-1k')
    parser.add_argument('--data_transforms', type=str, default='imagenet1k_mobile', help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
    parser.add_argument('--data_loader_workers', type=int, default=8,
                        help='number of workers for loading dataset (default: 8)')
    parser.add_argument('--test_only', type=bool, default=True, help='only test the dataset with pretained model')
    parser.add_argument('--pretained_checkpoint_path', type=str,
                        default='train_ckpt/AutoSlim-pretrained.ckpt',
                        help='The path of checkpoint for test-only or resume-train')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)
    device_num = 1
    rank_id = 0
    # dataset
    train_transforms, val_transforms = data_transforms(args)
    val_set = create_dataset(train_transforms,
                             val_transforms,
                             args,
                             dataset_path=args.dataset_path,
                             device_num=device_num,
                             rank_id=rank_id)[1]
    # load model
    print('Start loading model.')
    net = AutoSlimModel()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss_fn, optimizer=None, metrics={'acc'})
    load_checkpoint(args.pretained_checkpoint_path, net)
    print('Start evaluating.')
    acc = model.eval(val_set)
    print(acc)

if __name__ == "__main__":
    main()
