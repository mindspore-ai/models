# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Export mindir or air model for refinedet"""
import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.refinedet import refinedet_vgg16, refinedet_resnet101, RefineDetInferWithDecoder
from src.config import get_config
from src.box_utils import box_init

parser = argparse.ArgumentParser(description='RefineDet export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
parser.add_argument("--using_mode", type=str, default="refinedet_vgg16_320",
                    choices=("refinedet_vgg16_320", "refinedet_vgg16_512",
                             "refinedet_resnet101_320", "refinedet_resnet101_512"),
                    help="using mode, same as training.")
parser.add_argument("--file_name", type=str, default="refinedet", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR", "ONNX"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    config = get_config(args.using_mode, args.dataset)
    default_boxes = box_init(config)
    if config.model == "refinedet_vgg16":
        net = refinedet_vgg16(config=config, is_training=False)
    elif config.model == "refinedet_resnet101":
        net = refinedet_resnet101(config=config, is_training=False)
    else:
        raise ValueError(f'config.model: {config.model} is not supported')
    net = RefineDetInferWithDecoder(net, Tensor(default_boxes), config)

    param_dict = load_checkpoint(args.ckpt_file)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [args.batch_size, 3] + config.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mindspore.float32)
    export(net, input_array, file_name=args.file_name, file_format=args.file_format)
