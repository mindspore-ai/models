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
"""Export checkpoint file into air, onnx, mindir models."""

import argparse

import numpy as np
from mindspore import (
    context,
    export,
    load_checkpoint,
    load_param_into_net,
    Tensor,
)

import src.options.options as option
from src.network import create_model, IRN_loss

# Path to option YMAL file.
X2_TEST_YAML_FILE = "../src/options/test/test_IRN_x2.yml"
# Path to option YMAL file.
X4_TEST_YAML_FILE = "../src/options/test/test_IRN_x4.yml"

parser = argparse.ArgumentParser(description='IRN export.')
parser.add_argument(
    '--scale', type=int, default=4, choices=(2, 4),
    help='Rescaling Parameter.'
)
parser.add_argument('--device_id', type=int, default=0, help='Device id.',)
parser.add_argument('--checkpoint_path', type=str,
                    required=True, help='Checkpoint file path.',)
parser.add_argument('--file_name', type=str,
                    default='wrn-autoaugment', help='Output file name.',)
parser.add_argument(
    '--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'],
    default='MINDIR', help='Export format.',
)
parser.add_argument(
    '--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'],
    default='Ascend', help='Device target.',
)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.scale == 2:
        opt = option.parse(X2_TEST_YAML_FILE, args.dataset_GT_path,
                           args.dataset_LQ_path, is_train=False)
    elif args.scale == 4:
        opt = option.parse(X4_TEST_YAML_FILE, args.dataset_GT_path,
                           args.dataset_LQ_path, is_train=False)
    else:
        raise ValueError("Unsupported scale.")

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
    )
    if args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)

    # loading options for model
    opt = option.dict_to_nonedict(opt)

    # define net
    net = create_model(opt)

    param_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net, param_dict)

    loss = IRN_loss(net, opt)

    image = Tensor(np.ones((1, 3, 32, 32), np.float32))
    export(
        loss, image,
        file_name=args.file_name,
        file_format=args.file_format,
    )
