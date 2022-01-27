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
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src import init_model

parser = argparse.ArgumentParser(description='resnet export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, default='resnet50-300_93.ckpt', help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="resnet50_imagenet", help="output file name.")
parser.add_argument('--width', type=int, default=128, help='input width')
parser.add_argument('--height', type=int, default=256, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU", "CPU"])
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--outdir', type=str, default='')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

if __name__ == '__main__':
    net = init_model(name=args.arch, num_classes=751, loss='softmax and metric', aligned=True, is_train=False)
    net.set_train(False)
    assert args.ckpt_file is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
    print('export sucess')
