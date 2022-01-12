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

import argparse
import numpy as np
from mindspore import Tensor, export, load_checkpoint
import mindspore as ms
from model_utils.options import Options_310
from src.network import AutoEncoder

parser = argparse.ArgumentParser('export')
parser.add_argument('--ckpt_path', type=str, default='./ssim_autocoder.ckpt', help='ckpt data dir')
parser.add_argument('--file_name', type=str, default='AESSIM', help='mindir file name')
args = parser.parse_args()
cfg = Options_310().opt

def model_export():
    net = AutoEncoder(cfg)
    load_checkpoint(args.ckpt_path, net=net)
    net.set_train(False)
    channel = 1 if cfg["grayscale"] else 3
    input_size = cfg["data_augment"]["crop_size"]
    input_data = Tensor(np.ones([1, channel, input_size, input_size]), ms.float32)
    export(net, input_data, file_name=args.file_name, file_format="MINDIR")


if __name__ == '__main__':
    model_export()
