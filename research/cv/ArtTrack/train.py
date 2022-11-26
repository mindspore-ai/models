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
from mindspore import context
from mindspore.common import set_seed

from mindspore.context import ParallelMode
from mindspore import Model, nn
from mindspore.communication import get_group_size, get_rank, init
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor

from config import check_config
from src.args_util import command, create_arg_parser, TARGET_COCO_MULTI, TARGET_MPII_SINGLE
from src.dataset.util import create_dataset
from src.model.pose import PoseNet, PoseNetTotalLoss
from src.tool.decorator import process_cfg
from src.dataset.mpii import MPII
set_seed(1)

@command
def train(parser, args, cfg):
    """
    start train
    """

    @process_cfg
    def gpu_set(cfg, dataset_class, train_net):
        context.set_context(**cfg.context)
        group_size = None
        rank_id = None
        if hasattr(cfg, 'parallel_context') and cfg.parallel_context is not None:
            init()
            rank_id = get_rank()
            group_size = get_group_size()
            context.set_auto_parallel_context(
                device_num=group_size, **cfg.parallel_context)
            context.set_auto_parallel_context(parameter_broadcast=True)
        dataset = dataset_class(cfg)
        dataset = create_dataset(cfg.dataset.type, dataset, cfg.dataset.shuffle, cfg.dataset.batch_size,
                                 parallel=cfg.dataset.parallel, train=True, num_shards=group_size, rank_id=rank_id)
        model = Model(train_net)
        steps_per_epoch = dataset.get_dataset_size()
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=10)
        ckpt_dir = cfg.get('ckpt_dir', 'ckpt')
        ckpt_dir = ckpt_dir if rank_id is None else os.path.join(
            ckpt_dir, 'rank_%s' % str(rank_id))
        ckpt_cb = ModelCheckpoint(prefix=cfg.get('ckpt_prefix', 'arttrack'), directory=ckpt_dir,
                                  config=ckpt_config)
        callbacks = [TimeMonitor(data_size=steps_per_epoch),
                     LossMonitor(), ckpt_cb]
        model.train(cfg.epoch, dataset, callbacks=callbacks)

    def ascend_set(args, train_net, cfg=None):
        print("loading parse...")
        device_id = args.device_id
        if args.is_model_arts:
            import moxing as mox
            mox.file.copy_parallel(src_url=args.data_url,
                                   dst_url='/cache/data_tzh/')
        cfg = check_config(cfg, args)
        cfg.model_arts.GENERAL_RUN_DISTRIBUTE = args.run_distribute
        cfg.model_arts.IS_MODEL_ARTS = args.is_model_arts
        if cfg.model_arts.GENERAL_RUN_DISTRIBUTE or cfg.model_arts.IS_MODEL_ARTS:
            device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            save_graphs=False,
                            device_id=device_id)
        if cfg.model_arts.GENERAL_RUN_DISTRIBUTE:
            init()
            rank = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            rank = 0
            device_num = 1
        if cfg.model_arts.IS_MODEL_ARTS:
            mox.file.copy_parallel(src_url=args.data_url,
                                   dst_url=cfg.model_arts.CACHE_INPUT)
        dataset = MPII(cfg)
        dataset = create_dataset(cfg.dataset.type, dataset, cfg.dataset.shuffle, cfg.dataset.batch_size,
                                 parallel=cfg.dataset.parallel, train=True, num_shards=device_num, rank_id=rank,)
        dataset_size = dataset.get_dataset_size()

        time_cb = TimeMonitor(data_size=dataset_size)
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        config_ck = CheckpointConfig(
            save_checkpoint_steps=dataset_size, keep_checkpoint_max=5)

        if cfg.model_arts.GENERAL_RUN_DISTRIBUTE:
            prefix = 'multi_' + 'train_fastpose_' + \
                cfg.model_arts.VERSION + '_' + os.getenv('DEVICE_ID')
        else:
            prefix = 'single_' + 'train_fastpose_' + cfg.model_arts.VERSION

        if cfg.model_arts.IS_MODEL_ARTS:
            directory = cfg.model_arts.CACHE_OUTPUT + \
                'device_' + os.getenv('DEVICE_ID')
        elif cfg.model_arts.GENERAL_RUN_DISTRIBUTE:
            directory = cfg.under_line.CKPT_path + \
                'device_' + os.getenv('DEVICE_ID')
        else:
            directory = cfg.under_line.CKPT_path + 'device'
        ckpoint_cb = ModelCheckpoint(
            prefix=prefix, directory=directory, config=config_ck)
        if int(os.getenv('DEVICE_ID')) == 0:
            cb.append(ckpoint_cb)
        model = Model(train_net)
        print("************ Start training now ************")
        model.train(15, dataset, callbacks=cb)
        if cfg.model_arts.IS_MODEL_ARTS:
            mox.file.copy_parallel(
                src_url=cfg.model_arts.CACHE_OUTPUT, dst_url=args.train_url)

    net = PoseNet(cfg=cfg)
    loss = PoseNetTotalLoss(net, cfg)
    optimizer = nn.SGD(loss.trainable_params(),
                       learning_rate=nn.dynamic_lr.piecewise_constant_lr(cfg.multi_step[1], cfg.multi_step[0]))
    train_net = nn.TrainOneStepCell(loss, optimizer)
    train_net.set_train()
    if hasattr(cfg, 'load_ckpt') and os.path.exists(cfg.load_ckpt):
        mindspore.load_checkpoint(cfg.load_ckpt, net=train_net)

    if args.device_target == "GPU":
        if args.target == TARGET_MPII_SINGLE:
            gpu_set(cfg, MPII, train_net)
        elif args.target == TARGET_COCO_MULTI:
            from src.dataset import MSCOCO
            gpu_set(cfg, MSCOCO, train_net)
        else:
            parser.print_help()
    else:
        ascend_set(args, train_net)

def main():
    parser = create_arg_parser()['train']
    args = parser.parse_args(sys.argv[1:])
    train(parser, args)

if __name__ == '__main__':
    main()
