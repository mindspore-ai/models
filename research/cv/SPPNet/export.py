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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.config import sppnet_mult_cfg, sppnet_single_cfg, zfnet_cfg
from src.sppnet import SppNet

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--device_target', type=str, default="Ascend",
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument('--export_model', type=str, default='sppnet_single', help='chose the training model',
                    choices=['sppnet_single', 'sppnet_mult', 'zfnet'])
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

if __name__ == '__main__':

    if args.train_model == "zfnet":
        cfg = zfnet_cfg
        network = SppNet(cfg.num_classes, train_model=args.train_model)

    elif args.train_model == "sppnet_single":
        cfg = sppnet_single_cfg
        network = SppNet(cfg.num_classes, train_model=args.train_model)

    else:
        cfg = sppnet_mult_cfg
        network = SppNet(cfg.num_classes, train_model=args.train_model)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)
    input_arr = Tensor(np.zeros([args.batch_size, 3, cfg.image_height, cfg.image_width]), ms.float32)
    export(network, input_arr, file_name=args.train_model, file_format=args.file_format)
