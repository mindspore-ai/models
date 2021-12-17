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

"""Transfer data format"""

import argparse

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.box_utils import default_boxes
from src.config import config
from src.ssd_resnet34 import SSDInferWithDecoder


parser = argparse.ArgumentParser(description='SSD export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="ssd", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["MINDIR", "ONNX"], default="MINDIR",
                    help='file format')
parser.add_argument("--device_target", type=str, choices=["GPU", "CPU"], default="GPU",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target in ['GPU', 'Ascend']:
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    if config.model != "ssd-resnet34":
        raise ValueError(f'config.model: {config.model} is not supported')

    net = SSDInferWithDecoder(Tensor(default_boxes), config)

    param_dict = load_checkpoint(args.ckpt_file)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [args.batch_size, 3] + config.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mstype.float32)
    export(net, input_array, file_name=args.file_name, file_format=args.file_format)
