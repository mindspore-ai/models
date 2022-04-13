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
"""export"""
import argparse
import mindspore
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
import numpy as np
from src.network_G import GNet4_4_Train, GNet4_4_last, GNetNext_Train, GNetNext_Last


def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore PGAN training')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')
    parser.add_argument('--device_target', type=str, required=True, choices=['Ascend', 'GPU'], help='Device target')
    parser.add_argument('--checkpoint_g', type=str, default='ckpt', help='checkpoint dir of PGAN')
    args = parser.parse_args()
    context.set_context(device_id=args.device_id, mode=context.GRAPH_MODE, device_target=args.device_target)
    return args


def buildNoiseData(n_samples):
    """buildNoiseData

    Returns:
        output.
    """
    inputLatent = np.random.randn(n_samples, 512)
    inputLatent = mindspore.Tensor(inputLatent, mindspore.float32)
    return inputLatent


def main():
    """main"""
    args = preLauch()
    scales = [4, 8, 16, 32, 64, 128]
    depth = [512, 512, 512, 512, 256, 128]
    for scale_index, scale in enumerate(scales):
        if scale == 4:
            avg_gnet = GNet4_4_Train(512, depth[scale_index], leakyReluLeak=0.2, dimOutput=3)
        elif scale == 8:
            last_avg_gnet = GNet4_4_last(avg_gnet)
            avg_gnet = GNetNext_Train(depth[scale_index], last_Gnet=last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
        else:
            last_avg_gnet = GNetNext_Last(avg_gnet)
            avg_gnet = GNetNext_Train(depth[scale_index], last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)

    netG = avg_gnet
    param_G = load_checkpoint(args.checkpoint_g)
    load_param_into_net(netG, param_G)
    netG.set_train(False)
    inputNoise = buildNoiseData(64)
    export(netG, inputNoise, mindspore.Tensor(0.0, mindspore.float32), file_name="PGAN", file_format="MINDIR")
    print("PGAN exported")


if __name__ == '__main__':
    main()
