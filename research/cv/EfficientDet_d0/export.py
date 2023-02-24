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
"""Export EfficientDet on Coco"""
import argparse
import numpy as np
from src.backbone import EfficientDetBackbone
from src.config import config
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

parser = argparse.ArgumentParser(description='EfficientDet')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default="MINDIR")

args_opt = parser.parse_args()


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                        save_graphs=False)
    network = EfficientDetBackbone(config.num_classes, 0, False, False)

    # load checkpoint
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        not_load_param, _ = load_param_into_net(network, param_dict)
        if not_load_param:
            raise ValueError("Load param into network fail!")
    # export network
    print("============== Starting export ==============")
    inputs = Tensor(np.ones([1, 3, 512, 512]).astype(np.float32))
    export(network, inputs, file_name="EfficientDet_b0", file_format=args_opt.file_format)
    print("============== End export ==============")
