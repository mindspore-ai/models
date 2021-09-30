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
Process the test set with the .ckpt model in turn.
"""
import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.config import cifar10_cfg
from src.dataset import create_dataset_cifar10

from src.modeling_ms import VisionTransformer
import src.net_config as configs

set_seed(1)

parser = argparse.ArgumentParser(description='vit_base')
parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10'],
                    help='dataset name.')
parser.add_argument('--sub_type', type=str, default='ViT-B_16',
                    choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'testing'])
parser.add_argument('--checkpoint_path', type=str, default='./ckpt_0', help='Checkpoint file path')
parser.add_argument('--id', type=int, default=0, help='Device id')
args_opt = parser.parse_args()


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss_ = self.ce(logit, label)
        return loss_


if __name__ == '__main__':
    CONFIGS = {'ViT-B_16': configs.get_b16_config,
               'ViT-B_32': configs.get_b32_config,
               'ViT-L_16': configs.get_l16_config,
               'ViT-L_32': configs.get_l32_config,
               'ViT-H_14': configs.get_h14_config,
               'R50-ViT-B_16': configs.get_r50_b16_config,
               'testing': configs.get_testing}
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args_opt.id)
    if args_opt.dataset_name == "cifar10":
        cfg = cifar10_cfg
        net = VisionTransformer(CONFIGS[args_opt.sub_type], num_classes=cfg.num_classes)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        opt = nn.Momentum(net.trainable_params(), 0.01, cfg.momentum, weight_decay=cfg.weight_decay)
        dataset = create_dataset_cifar10(cfg.val_data_path, 1, False)
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        print("load checkpoint from [{}].".format(args_opt.checkpoint_path))
        load_param_into_net(net, param_dict)
        net.set_train(False)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
    else:
        raise ValueError("dataset is not support.")

    acc = model.eval(dataset)
    print(f"model's accuracy is {acc}")
