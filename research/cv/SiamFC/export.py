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
"""export checkpoint file into models"""
import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, export, load_param_into_net
from src.alexnet import SiameseAlexNet

parser = argparse.ArgumentParser(description='siamfc export')
parser.add_argument("--device_id", type=int, default=7, help="Device id")
parser.add_argument('--model_path', default='/root/models/siamfc_{}.ckpt/SiamFC_177-47_6650.ckpt'
                    , type=str, help='eval one special video')
parser.add_argument('--file_name_export1', type=str, default='/root/SiamFC/models1',
                    help='SiamFc output file name.')
parser.add_argument('--file_name_export2', type=str, default='/root/SiamFC/models2',
                    help='SiamFc output file name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__  == "__main__":
    net1 = SiameseAlexNet(train=False)
    load_param_into_net(net1, load_checkpoint(args.model_path), strict_load=True)
    net1.set_train(False)
    net2 = SiameseAlexNet(train=False)
    load_param_into_net(net2, load_checkpoint(args.model_path), strict_load=True)
    net2.set_train(False)
    input_data_exemplar1 = Tensor(np.zeros([1, 3, 127, 127]), ms.float32)
    input_data_instance1 = Tensor(np.zeros(1), ms.float32)
    input_data_exemplar2 = Tensor(np.ones([1, 256, 6, 6]), ms.float32)
    input_data_instance2 = Tensor(np.ones([1, 3, 255, 255]), ms.float32)
    input1 = [input_data_exemplar1, input_data_instance1]
    input2 = [input_data_exemplar2, input_data_instance2]
    export(net1, *input1, file_name=args.file_name_export1, file_format=args.file_format)
    export(net2, *input2, file_name=args.file_name_export2, file_format=args.file_format)
    print("--   complete    --")