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

from mindspore import dtype as mstype
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

from src.models import TSN

parser = argparse.ArgumentParser(description='TSN')
parser.add_argument('--ckpt_path', type=str, default="")
parser.add_argument('--dataset', type=str, default="ucf101", choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--modality', type=str, default="Flow", choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'), help='run platform')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument("--file_name", type=str, default="tsn_", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, device_id=args.device_id)

if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.dataset)

if __name__ == '__main__':

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
        args.dropout = 0.3
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    images = args.test_segments * args.test_crops
    net = TSN(num_class, 1, args.modality, base_model=args.arch,\
         consensus_type=args.crop_fusion_type, dropout=args.dropout)

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([images, length, args.input_size, args.input_size]), mstype.float32)
    export(net, input_arr, file_name=args.file_name+args.modality, file_format=args.file_format)
