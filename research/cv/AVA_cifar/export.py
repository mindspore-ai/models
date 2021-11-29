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
"""export."""

import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, export

from src.config import get_config
from src.cifar_resnet import resnet18, resnet50, resnet101
from src.knn_eval import FeatureCollectCell310

parser = argparse.ArgumentParser(description="export")
parser.add_argument("--device_id", type=int, default=0,
                    help="Device id, default is 0.")
parser.add_argument("--device_target", type=str,
                    default="Ascend", help="Device target")
parser.add_argument("--load_ckpt_path", type=str, default="",
                    help="path to load pretrain model checkpoint")
parser.add_argument("--network", type=str, default="resnet18", choices=['resnet18', 'resnet50', 'resnet101'],
                    help="network architecture")
parser.add_argument("--file_name", type=str, default="ava_cifar", help="model file name")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['AIR', 'MINDIR'])
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    temp_path = ''

    device_id = args_opt.device_id
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=device_id)

    if args_opt.network == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims,
                          training_mode=True, use_MLP=config.use_MLP)
    elif args_opt.network == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims,
                          training_mode=True, use_MLP=config.use_MLP)
    elif args_opt.network == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims,
                           training_mode=True, use_MLP=config.use_MLP)
    else:
        raise "net work config error!!!"

    load_checkpoint(args_opt.load_ckpt_path, net=resnet)
    print("load ckpt from {}".format(args_opt.load_ckpt_path))
    eval_network = FeatureCollectCell310(resnet)
    bs = config.batch_size

    inputs = Tensor(np.random.uniform(low=0, high=1.0, size=(bs, 3, 32, 32)).astype(np.float32))
    export(eval_network, inputs, file_name=args_opt.file_name, file_format=args_opt.file_format)
