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

from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.resnet import resnet18, resnet50, resnet101
from src.network_define_eval import EvalCell310

parser = argparse.ArgumentParser(description="export")
parser.add_argument("--device_id", type=int, default=0,
                    help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1,
                    help="Use device nums, default is 1.")
parser.add_argument('--device_target', type=str,
                    default="Ascend", help='Device target')
parser.add_argument('--ckpt_path', type=str, default="",
                    help='model checkpoint path')
parser.add_argument("--model_arch", type=str, default="resnet18",
                    choices=['resnet18', 'resnet50', 'resnet101'], help='model architecture')
parser.add_argument("--classes", type=int, default=10, help='class number')
parser.add_argument("--file_name", type=str, default="ava_hpa", help='model name')
parser.add_argument("--file_format", type=str, default="MINDIR",
                    choices=['AIR', 'MINDIR'], help='model format')

args_opt = parser.parse_args()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)
    ckpt_path = args_opt.ckpt_path

    if args_opt.model_arch == 'resnet18':
        resnet = resnet18(pretrain=False, classes=args_opt.classes)
    elif args_opt.model_arch == 'resnet50':
        resnet = resnet50(pretrain=False, classes=args_opt.classes)
    elif args_opt.model_arch == 'resnet101':
        resnet = resnet101(pretrain=False, classes=args_opt.classes)
    else:
        raise "Unsupported net work!"
    param_dict = load_checkpoint(args_opt.ckpt_path)
    load_param_into_net(resnet, param_dict)

    bag_size_for_eval = 20
    image_shape = (224, 224)
    input_shape = (bag_size_for_eval, 3) + image_shape

    test_network = EvalCell310(resnet)
    input_data0 = Tensor(np.random.uniform(low=0, high=1.0, size=input_shape).astype(np.float32))
    export(test_network, input_data0, file_name=args_opt.file_name, file_format=args_opt.file_format)
