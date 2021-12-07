# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""export checkpoint file into AIR MINDIR ONNX models"""
import argparse
import ast
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

from src.config import config_gpu as cfg
from src.shufflenetv2 import ShuffleNetV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='checkpoint export')
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument('--width', type=int, default=224, help='input width')
    parser.add_argument('--height', type=int, default=224, help='input height')
    parser.add_argument("--file_name", type=str, default="shufflenetv2", help="output file name.")
    parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR",
                        help="file format")
    parser.add_argument("--device_target", type=str, default="GPU",
                        choices=["Ascend", "GPU", "CPU"],
                        help="device where the code will be implemented (default: GPU)")
    parser.add_argument('--overwrite_config', type=ast.literal_eval, default=False,
                        help='whether to overwrite the config according to the arguments')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')

    args = parser.parse_args()
    if args.overwrite_config:
        cfg.num_classes = args.num_classes

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend" or args.device_target == "GPU":
        context.set_context(device_id=args.device_id)

    net = ShuffleNetV2(n_class=cfg.num_classes)
    ckpt = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, ckpt)
    net.set_train(False)

    input_data = Tensor(np.ones([args.batch_size, 3, args.height, args.width]), ms.float32)
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)
