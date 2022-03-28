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

import time
import argparse

from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.logger import get_logger
from src.loss.loss import get_loss
from src.models.vit import FineTuneVit
from src.datasets.dataset import get_dataset
from src.monitors.monitor import StateMonitor
from src.lr.lr_generator import LearningRate
from src.lr.lr_decay import param_groups_lrd
from src.trainer.trainer import create_train_one_step
from src.models.eval_engine import get_eval_engine
from src.model_utils.moxing_adapter import moxing_wrapper
from src.helper import parse_with_config, str2bool, cloud_context_init


@moxing_wrapper()
def main(args):
    local_rank, device_num = cloud_context_init(seed=args.seed, use_parallel=args.use_parallel,
                                                context_config=args.context, parallel_config=args.parallel)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))

    # train dataset
    train_dataset = get_dataset(args)
    data_size = train_dataset.get_dataset_size()
    new_epochs = args.epoch
    if args.per_step_size:
        new_epochs = int((data_size / args.per_step_size) * args.epoch)
    else:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}， sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    # evaluation dataset
    eval_dataset = get_dataset(args, is_train=False)

    net = FineTuneVit(batch_size=args.batch_size, patch_size=args.patch_size,
                      image_size=args.image_size, dropout=args.dropout,
                      num_classes=args.num_classes, **args.model_config)
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    if args.start_learning_rate is None:
        args.start_learning_rate = (args.base_lr * args.device_num * args.batch_size) / 256
    # define lr_schedule
    lr_schedule = LearningRate(
        args.start_learning_rate, args.end_learning_rate,
        args.epoch, args.warmup_epochs, data_size
    )

    group_params = net.trainable_params()
    # load finetune ckpt
    if args.use_ckpt:
        # layer-wise lr decay
        no_weight_decay_list = net.no_weight_decay()
        group_params = param_groups_lrd(net, weight_decay=args.weight_decay,
                                        no_weight_decay_list=no_weight_decay_list)
        params_dict = load_checkpoint(args.use_ckpt)
        net_not_load = net.init_weights(params_dict)
        args.logger.info(f"===============net_not_load================{net_not_load}")

    # define optimizer
    optimizer = nn.AdamWeightDecay(group_params,
                                   learning_rate=lr_schedule,
                                   weight_decay=args.weight_decay,
                                   beta1=args.beta1,
                                   beta2=args.beta2)

    # define loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    vit_loss = get_loss(loss_name=args.loss_name, args=args)

    # Build train network
    net_with_loss = nn.WithLossCell(net, vit_loss)
    net_with_train = create_train_one_step(args, net_with_loss, optimizer, log=args.logger)

    # define callback
    state_cb = StateMonitor(data_size=args.per_step_size,
                            tot_batch_size=args.batch_size * device_num,
                            eval_interval=args.eval_interval,
                            eval_offset=args.eval_offset,
                            eval_engine=eval_engine,
                            logger=args.logger.info)
    callback = [state_cb,]
    # model config
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq,
                                     keep_checkpoint_max=1,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix,
                                     directory=args.save_dir,
                                     config=config_ck)
        callback += [ckpoint_cb,]

    # define Model and begin training
    model = Model(net_with_train, metrics=eval_engine.metric,
                  eval_network=eval_engine.eval_network)

    eval_engine.set_model(model)
    t0 = time.time()
    # equal to model._init(dataset, sink_size=per_step_size)
    eval_engine.compile(sink_size=args.per_step_size)
    t1 = time.time()
    args.logger.info('compile time used={:.2f}s'.format(t1 - t0))
    model.train(new_epochs, train_dataset, callbacks=callback,
                dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)
    last_metric = 'last_metric[{}]'.format(state_cb.best_acc)
    args.logger.info(last_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_path', default="/home/work/user-job-dir/mae_mindspore/config/finetune-vit-base-p32-448.yaml",
        help='YAML config files')
    parser.add_argument(
        '--use_parallel', default=False, type=str2bool, help='use parallel config.')

    args_ = parse_with_config(parser)
    if args_.eval_offset < 0:
        args_.eval_offset = args_.epoch % args_.eval_interval

    main(args_)
