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
eval.
"""
import os
import argparse
from mindspore import context
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.ghostnet import ghostnet_1x, ghostnet_nose_1x
from src.ghostnet600 import ghostnet_600m
from src.config import config

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--num_classes', type=int, default=5, help='Classes number')
parser.add_argument('--device_target', type=str, default='CPU', help='Device platform')
parser.add_argument('--checkpoint_path', type=str, default='./ckpt/ghostnet_1x_2-500_21.ckpt',
                    help='Checkpoint file path')
parser.add_argument('--data_url', type=str, default='./dataset/', help='Dataset path')
args_opt = parser.parse_args()


if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=device_id,
                        save_graphs=False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    if 'ghostnet_1x' in args_opt.checkpoint_path:
        net = ghostnet_1x(num_classes=args_opt.num_classes)
    elif 'ghostnet_nose_1x' in args_opt.checkpoint_path:
        net = ghostnet_nose_1x(num_classes=args_opt.num_classes)
    elif 'ghostnet600M' in args_opt.checkpoint_path:
        net = ghostnet_600m(num_classes=args_opt.num_classes)

    dataset = create_dataset(dataset_path=args_opt.data_url, do_train=False, batch_size=config.batch_size,
                             num_parallel_workers=None)
    step_size = dataset.get_dataset_size()

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
