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
"""export"""
import argparse
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.config import common_config, VeRi_test
from src.VehicleNet_resnet50 import VehicleNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VehicleNet')
    parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"],
                        default="Ascend", help="device target")
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument("--ckpt_url", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--file_name", type=str, default="vehiclenet", help="output file name.")
    parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"],
                        default='MINDIR', help='file format')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    cfg = common_config
    dataset_cfg = VeRi_test

    net = VehicleNet(dataset_cfg.num_classes)
    net.classifier.classifier = nn.SequentialCell()

    param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, 384, 384]).astype(np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)
