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

"""export ckpt to model"""

import argparse

import numpy as np

import mindspore
from mindspore.train.serialization import export
from mindspore import context, Tensor
from mindspore.common import set_seed
from src.config import yolact_plus_resnet50_config as cfg
from src.yolact.yolactpp import Yolact

parser = argparse.ArgumentParser(description="Yolact export")
parser.add_argument("--device_id", type=int, default=2, help="Device id")
parser.add_argument("--ckpt_file", type=str, default="./yolact-20_619.ckpt", help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="Yolact", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()
set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    net = Yolact()
    net.set_train(False)
    img = Tensor(np.ones([1, 3, cfg['img_height'], cfg['img_width']]), mindspore.float32)
    export(net, img, file_name=args.file_name, file_format=args.file_format)
