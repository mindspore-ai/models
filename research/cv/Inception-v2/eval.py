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
"""evaluate_imagenet"""
import argparse
import ast
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config_gpu, config_ascend, config_cpu
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inception_v2 import inception_v2_base
from src.loss import CrossEntropy_Val

CFG_DICT = {
    "Ascend": config_ascend,
    "GPU": config_gpu,
    "CPU": config_cpu,
}

DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}


def run_eval():
    """run evaluation"""
    parser = argparse.ArgumentParser(description='image classification evaluation')
    parser.add_argument("--data_url", type=str, help='data path for eval')
    parser.add_argument("--train_url", type=str, help='log')
    parser.add_argument("--run_online", type=ast.literal_eval)
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint of inception-v2 (Default: None)')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')
    args_opt = parser.parse_args()
    cfg = CFG_DICT[args_opt.platform]
    cfg.work_nums = 1
    if args_opt.run_online:
        import moxing as mox

        cfg.ckpt_path = "/cache/checkpoint_inceptionv2/checkpoint.ckpt"
        Imagenet_root = "/cache/data_eval_url"
        mox.file.copy_parallel(args_opt.data_url, Imagenet_root)
        mox.file.copy_parallel(args_opt.checkpoint, cfg.ckpt_path)
    else:
        cfg.ckpt_path = args_opt.checkpoint
        Imagenet_root = args_opt.data_url
    if args_opt.platform == 'Ascend':
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
    create_dataset = DS_DICT[cfg.ds_type]
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform)
    net = inception_v2_base(num_classes=cfg.num_classes)
    ckpt = load_checkpoint(cfg.ckpt_path)
    load_param_into_net(net, ckpt)
    net.set_train(False)
    cfg.rank = 0
    cfg.group_size = 1
    # eval dataset
    root = os.path.join(Imagenet_root, 'val')
    dataset = create_dataset(root, cfg, False)
    loss = CrossEntropy_Val(smooth_factor=0.1, num_classes=cfg.num_classes)
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss, optimizer=None, metrics=eval_metrics)
    metrics = model.eval(dataset, dataset_sink_mode=cfg.ds_sink_mode)
    print("metric: ", metrics)


if __name__ == '__main__':
    run_eval()
