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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse

import numpy as np
import mindspore.common.dtype as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from config import check_config

from src.model.pose import PoseNet, PoseNetTestExport

def parse_args():
    parser = argparse.ArgumentParser(description="Export Mindir or Air")
    parser.add_argument('--device_id', required=True, default=1,
                        type=int, help='Location of training outputs.')
    parser.add_argument("--ckpt_url", default="art_track.ckpt",
                        help="Checkpoint file path.")
    parser.add_argument("--file_name", type=str, default="art_track", help="output file name.")
    parser.add_argument('--file_format', type=str, choices=["MINDIR", "AIR"], default='MINDIR', help='file format')
    parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"],
                        default="Ascend", help="device target")
    args = parser.parse_args()
    return args
def main(cfg=None):
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    cfg = check_config(cfg, args)
    net = PoseNet(cfg=cfg)
    test_net = PoseNetTestExport(net, cfg)
    param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(test_net, param_dict)
    input_arr = Tensor(np.ones([1, 3, 415, 320]), ms.float32)
    export(test_net, input_arr, file_name=args.file_name, file_format=args.file_format)
if __name__ == '__main__':
    main()
