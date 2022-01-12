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
import os
import sys

import mindspore
import mindspore.context as ctx
from mindspore import Model, nn
from mindspore.communication import get_group_size, get_rank, init
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor

from src.args_util import command, create_arg_parser, TARGET_COCO_MULTI, TARGET_MPII_SINGLE
from src.dataset.util import create_dataset
from src.model.pose import PoseNet, PoseNetTotalLoss
from src.tool.decorator import process_cfg


@command
def train(parser, args, cfg):
    if args.target == TARGET_MPII_SINGLE:
        from src.dataset import MPII
        start_train(cfg, MPII)
    elif args.target == TARGET_COCO_MULTI:
        from src.dataset import MSCOCO
        start_train(cfg, MSCOCO)
    else:
        parser.print_help()


@process_cfg
def start_train(cfg, dataset_class):
    """
    start train
    """
    ctx.set_context(**cfg.context)
    group_size = None
    rank_id = None
    if hasattr(cfg, 'parallel_context') and cfg.parallel_context is not None:
        init()
        rank_id = get_rank()
        group_size = get_group_size()
        ctx.set_auto_parallel_context(device_num=group_size, **cfg.parallel_context)
        ctx.set_auto_parallel_context(parameter_broadcast=True)
    dataset = dataset_class(cfg)
    dataset = create_dataset(cfg.dataset.type, dataset, cfg.dataset.shuffle, cfg.dataset.batch_size,
                             parallel=cfg.dataset.parallel, train=True, num_shards=group_size, rank_id=rank_id)
    net = PoseNet(cfg=cfg)
    loss = PoseNetTotalLoss(net, cfg)
    optimizer = nn.SGD(loss.trainable_params(),
                       learning_rate=nn.dynamic_lr.piecewise_constant_lr(cfg.multi_step[1], cfg.multi_step[0]))
    train_net = nn.TrainOneStepCell(loss, optimizer)
    train_net.set_train()
    if hasattr(cfg, 'load_ckpt') and os.path.exists(cfg.load_ckpt):
        mindspore.load_checkpoint(cfg.load_ckpt, net=train_net)
    model = Model(train_net)
    steps_per_epoch = dataset.get_dataset_size()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=10)
    ckpt_dir = cfg.get('ckpt_dir', 'ckpt')
    ckpt_dir = ckpt_dir if rank_id is None else os.path.join(ckpt_dir, 'rank_%s' % str(rank_id))
    ckpt_cb = ModelCheckpoint(prefix=cfg.get('ckpt_prefix', 'arttrack'), directory=ckpt_dir,
                              config=ckpt_config)
    callbacks = [TimeMonitor(data_size=steps_per_epoch), LossMonitor(), ckpt_cb]
    model.train(cfg.epoch, dataset, callbacks=callbacks, dataset_sink_mode=False)


def main():
    parser = create_arg_parser()['train']
    args = parser.parse_args(sys.argv[1:])
    train(parser, args)


if __name__ == '__main__':
    main()
