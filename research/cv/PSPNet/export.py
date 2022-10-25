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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np
import src.utils.functions_args as fa
from src.model import pspnet
import mindspore.common.dtype as dtype
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

parser = argparse.ArgumentParser(description='maskrcnn export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--yaml_path", type=str, required=True, default='./config/voc2012_pspnet50.yaml',
                    help='yaml file path')
parser.add_argument("--ckpt_file", type=str, required=True, default='./checkpoints/voc/ADE-50_1063.ckpt',
                    help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="PSPNet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target (default: Ascend)')
parser.add_argument("--project_path", type=str, default='/root/PSPNet/',
                    help="project_path,default is /root/PSPNet/")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    config_path = args.yaml_path
    cfg = fa.load_cfg_from_cfg_file(config_path)

    net = pspnet.PSPNet(
        feature_size=cfg.feature_size,
        num_classes=cfg.classes,
        backbone=cfg.backbone,
        pretrained=False,
        pretrained_path="",
        aux_branch=False,
        deep_base=True
    )
    param_dict = load_checkpoint(args.ckpt_file)

    load_param_into_net(net, param_dict, strict_load=True)
    net.set_train(False)

    img = Tensor(np.ones([args.batch_size, 3, 473, 473]), dtype.float32)
    print("################## Start export ###################")
    export(net, img, file_name=args.file_name, file_format=args.file_format)
    print("################## Finish export ###################")
