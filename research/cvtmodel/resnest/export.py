# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""export checkpoint file into air, onnx, mindir models
   Suggest run as python export.py --backbone [backbone] --ckpt_path [ckpt_path]
                                   --file_name [file_name] --file_format [file format]
"""
import argparse
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='post process for 310 inference')
parser.add_argument("--backbone", type=str, required=True, default="resnest50", help="model backbone")
parser.add_argument("--ckpt_path", type=str, required=True, help="checkpoint file path")
parser.add_argument("--file_name", type=str, default="resnest50", help="file name")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=["MINDIR", "AIR"], help="file format")
parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"], help="device target")
parser.add_argument("--device_id", type=int, default=0, help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

def model_export():
    '''export main function'''
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)
    if args.backbone == "resnest50":
        from src.resnest50 import MainModel
        image_size = 224
    elif args.backbone == "resnest101":
        from src.resnest101 import MainModel
        image_size = 256
    elif args.backbone == "resnest200":
        from src.resnest200 import MainModel
        image_size = 320
    elif args.backbone == "resnest269":
        from src.resnest269 import MainModel
        image_size = 416

    net = MainModel()

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([1, 3, image_size, image_size]), ms.float32)
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)

if __name__ == '__main__':
    model_export()
