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
##############export checkpoint file into air, onnx or mindir model#################
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.modeling_ms import VisionTransformer
import src.net_config as configs

parser = argparse.ArgumentParser(description='vit_base export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument('--sub_type', type=str, default='ViT-B_16',
                    choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'testing'])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="vit_base", help="output file name.")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target(default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':

    CONFIGS = {'ViT-B_16': configs.get_b16_config,
               'ViT-B_32': configs.get_b32_config,
               'ViT-L_16': configs.get_l16_config,
               'ViT-L_32': configs.get_l32_config,
               'ViT-H_14': configs.get_h14_config,
               'R50-ViT-B_16': configs.get_r50_b16_config,
               'testing': configs.get_testing}
    net = VisionTransformer(CONFIGS[args.sub_type], num_classes=10)

    assert args.ckpt_file is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
