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
"""export checkpoint file into air, onnx, mindir models"""
import os
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from mindspore.common import set_seed

from src.resnet import wide_resnet50_2
from src.fastflow import build_model
from src.config import get_arguments
from src.loss import NetWithLossCell, FastflowLoss

set_seed(1)

def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # export Device.
    parser.add_argument('--device_target', type=str, default='Ascend')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--ckpt_file', type=str, required=True,
                        help='checkpoint file path')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='MINDIR',
                        help='file format')

    return parser.parse_args()

if __name__ == '__main__':
    args = preLauch()
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE, save_graphs=False)
    context.set_context(device_id=args.device_id)

    assert args.ckpt_file != '', "args.ckpt_file is empty."

    # network
    feature_extractor = wide_resnet50_2()
    network = build_model(
        backbone=feature_extractor,
        flow_step=args.flow_step,
        im_resize=args.im_resize,
        conv3x3_only=args.conv3x3_only,
        hidden_ratio=args.hidden_ratio
        )

    # keep network auto_prefix is same as ckpt
    loss = FastflowLoss()
    _ = NetWithLossCell(network, loss)

    # load param into net
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)
    network.set_train(False)

    # get file name
    print("ckpt_file: ", args.ckpt_file)
    head_tail = os.path.split(args.ckpt_file)
    file_name = os.path.splitext(head_tail[-1])[0]
    print("file_name: ", file_name)

    # export model
    input_arr = Tensor(np.random.randn(1, 3, 256, 256).astype(np.float32))

    export(network, input_arr, file_name=file_name, file_format=args.file_format)
    print("==========Fastflow exported==========")
