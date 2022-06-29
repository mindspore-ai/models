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
##############export checkpoint file into air, onnx or mindir model#################
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, export, context, load_checkpoint, load_param_into_net

from src.models import FaceNetModelwithLoss

parser = argparse.ArgumentParser(description='FaceNet export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--net_name", type=str, default="facenet", help="network name.")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=args.device_id)

if __name__ == '__main__':

    assert args.ckpt_file is not None, "checkpoint_path is None."

    net = FaceNetModelwithLoss(num_classes=1001, margin=0.5)
    state_dict = load_checkpoint(ckpt_file_name=args.ckpt_file, net=net)
    print("Loading the trained models from ckpt")
    load_param_into_net(net, state_dict)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    input_arr2 = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, input_arr2, file_name=args.net_name, file_format=args.file_format)
