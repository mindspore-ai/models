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
######################## export DEM ########################
"""

import mindspore as ms
from src.set_parser import set_parser
from src.utils import backbone_cfg

import numpy as np

if __name__ == "__main__":
    # Set graph mode, device id
    args = set_parser()
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, \
                           device_target=args.device_target, \
                           device_id=args.device_id)

    # Initialize parameters
    save_ckpt = args.save_ckpt

    # Build network
    net = backbone_cfg(args)

    # Eval
    print("============== Starting Evaluating ==============")
    if args.train_mode == 'att':
        ms.load_checkpoint(save_ckpt, net)
        if args.dataset == 'AwA':
            input0 = ms.Tensor(np.zeros([args.batch_size, 85]), ms.float32)
        elif args.dataset == 'CUB':
            input0 = ms.Tensor(np.zeros([args.batch_size, 312]), ms.float32)
        ms.export(net, input0, file_name="DEM_att", file_format=args.file_format)
        print("Successfully convert to", args.file_format)
    elif args.train_mode == 'word':
        ms.load_checkpoint(save_ckpt, net)
        input0 = ms.Tensor(np.zeros([args.batch_size, 1000]), ms.float32)
        ms.export(net, input0, file_name="DEM_word", file_format=args.file_format)
        print("Successfully convert to", args.file_format)
    elif args.train_mode == 'fusion':
        ms.load_checkpoint(save_ckpt, net)
        input1 = ms.Tensor(np.zeros([args.batch_size, 85]), ms.float32)
        input2 = ms.Tensor(np.zeros([args.batch_size, 1000]), ms.float32)
        ms.export(net, input1, input2, file_name="DEM_fusion", file_format=args.file_format)
        print("Successfully convert to", args.file_format)
