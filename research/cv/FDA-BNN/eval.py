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
"""Inference Interface"""
import sys
import argparse
import logging

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore import context

from src.dataset import create_dataset_cifar10
from src.loss import LabelSmoothingCrossEntropy
from src.resnet import resnet20

from easydict import EasyDict as edict

root = logging.getLogger()
root.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data_path', type=str, default='/home/xyx/FDA-BNN',
                    metavar='DIR', help='path to dataset')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 10)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--ckpt', type=str, default='./fdabnn.ckpt',
                    help='model checkpoint to load')
parser.add_argument('--image_size', type=int, default=32,
                    help='input image size')

def main():
    """Main entrance for training"""
    args = parser.parse_args()
    print(sys.argv)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

    net = resnet20()
    cfg = edict({
        'image_height': args.image_size,
        'image_width': args.image_size,
    })
    cfg.batch_size = args.batch_size
    val_data_url = args.data_path
    val_dataset = create_dataset_cifar10(val_data_url, repeat_num=1, training=False, cifar_cfg=cfg)
    loss = LabelSmoothingCrossEntropy(smooth_factor=args.smoothing,
                                      num_classes=args.num_classes)
    loss.add_flags_recursive(fp32=True, fp16=False)
    eval_metrics = {'Validation-Loss': Loss(),
                    'Top1-Acc': Top1CategoricalAccuracy(),
                    'Top5-Acc': Top5CategoricalAccuracy()}
    ckpt = load_checkpoint(args.ckpt)
    load_param_into_net(net, ckpt)
    net.set_train(False)
    model = Model(net, loss, metrics=eval_metrics)
    metrics = model.eval(val_dataset, dataset_sink_mode=False)
    print(metrics)



if __name__ == '__main__':
    main()
