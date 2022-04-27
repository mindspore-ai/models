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
######################## train example ########################
python train.py --dataset_path = /YourDataPath
"""
import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor, export, load_checkpoint
from mindspore.common import set_seed

from src.autoslim_resnet_for_val import AutoSlimModel

set_seed(1)

def main():
    parser = argparse.ArgumentParser(description='AutoSlim MindSpore Exporting')
    # Define parameters
    # device
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='number of device which is chosen')

    # export parameters
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 256)')
    parser.add_argument('--export_model_name', type=str, default='autoslim.mindir', help='')
    parser.add_argument('--pretained_checkpoint_path', type=str,
                        default='./train_ckpt/AutoSlim-pretrained.ckpt',
                        help='The path of checkpoint for test-only or resume-train')
    parser.add_argument("--file_format", type=str, default="MINDIR", choices=["AIR", "ONNX", "MINDIR"], help="export")

    args = parser.parse_args()

    # Set graph mode, device id
    context.set_context(mode=context.GRAPH_MODE, \
                        device_target=args.device_target, \
                        device_id=args.device_id)

    # Build network
    net = AutoSlimModel()
    load_checkpoint(args.pretained_checkpoint_path, net)

    print("============== Starting Exporting ==============")
    input0 = Tensor(np.zeros([args.batch_size, 3, 224, 224]), mindspore.float32)
    export(net, input0, file_name=args.export_model_name, file_format=args.file_format)
    print("Successfully convert to", args.file_format)

if __name__ == "__main__":
    main()
