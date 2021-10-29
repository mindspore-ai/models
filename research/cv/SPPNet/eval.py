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
######################## eval sppnet example ########################
eval sppnet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt --device_id YourAscendId --train_model model
"""

import ast
import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from src.config import sppnet_mult_cfg, sppnet_single_cfg, zfnet_cfg
from src.dataset import create_dataset_imagenet
from src.sppnet import SppNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore SPPNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend",
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--test_model', type=str, default='sppnet_single', help='chose the training model',
                        choices=['sppnet_single', 'sppnet_mult', 'zfnet'])
    parser.add_argument('--data_path', type=str, default="", help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                            path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=ast.literal_eval,
                        default=True, help='dataset_sink_mode is False or True')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend. (Default: 0)')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(device_id=args.device_id)
    print("============== Starting Testing ==============")

    if args.test_model == "zfnet":
        cfg = zfnet_cfg
        ds_eval = create_dataset_imagenet(args.data_path, 'zfnet', cfg.batch_size, training=False)
        network = SppNet(cfg.num_classes, phase='test', train_model=args.test_model)

    elif args.test_model == "sppnet_single":
        cfg = sppnet_single_cfg
        ds_eval = create_dataset_imagenet(args.data_path, cfg.batch_size, training=False)
        network = SppNet(cfg.num_classes, phase='test', train_model=args.test_model)

    else:
        cfg = sppnet_mult_cfg
        ds_eval = create_dataset_imagenet(args.data_path, cfg.batch_size, training=False)
        network = SppNet(cfg.num_classes, phase='test', train_model=args.test_model)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    param_dict = load_checkpoint(args.ckpt_path)
    print("load checkpoint from [{}].".format(args.ckpt_path))
    load_param_into_net(network, param_dict)
    network.set_train(False)

    model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    result = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("result : {}".format(result))
