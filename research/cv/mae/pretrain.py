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

from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.datasets.imagenet import create_dataset
from src.models.mae_vit import PreTrainMAEVit
from src.monitors.monitor import LossMonitor

from src.helper import parse_with_config, str2bool, cloud_context_init
from src.logger import get_logger
from src.lr.lr_generator import LearningRate
from src.trainer.trainer import create_train_one_step
from src.model_utils.moxing_adapter import moxing_wrapper


@moxing_wrapper()
def main(args):
    local_rank, device_num = cloud_context_init(seed=args.seed, use_parallel=args.use_parallel,
                                                context_config=args.context, parallel_config=args.parallel)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))

    # train dataset
    dataset = create_dataset(args)
    data_size = dataset.get_dataset_size()
    new_epochs = args.epoch
    if args.per_step_size:
        new_epochs = int((data_size / args.per_step_size) * args.epoch)
    else:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}， sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    net = PreTrainMAEVit(batch_size=args.batch_size, patch_size=args.patch_size, image_size=args.image_size,
                         encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                         encoder_num_heads=args.encoder_num_heads, decoder_num_heads=args.decoder_num_heads,
                         encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim,
                         mlp_ratio=args.mlp_ratio, masking_ratio=args.masking_ratio)

    if args.start_learning_rate == 0.:
        args.start_learning_rate = (args.base_lr * args.device_num * args.batch_size) / 256
    # define lr_schedule
    lr_schedule = LearningRate(
        args.start_learning_rate, args.end_learning_rate,
        args.epoch, args.warmup_epochs, data_size
    )

    # define optimizer
    optimizer = nn.AdamWeightDecay(net.trainable_params(),
                                   learning_rate=lr_schedule,
                                   weight_decay=args.weight_decay,
                                   beta1=args.beta1,
                                   beta2=args.beta2)

    # load pretrain ckpt
    if args.use_ckpt:
        params_dict = load_checkpoint(args.use_ckpt)
        load_param_into_net(net, params_dict)
        load_param_into_net(optimizer, params_dict)

    # define model
    train_model = create_train_one_step(args, net, optimizer, log=args.logger)

    # define callback
    callback = [LossMonitor(log=args.logger),]

    # define ckpt config
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
    model = Model(train_model)
    model.train(new_epochs, dataset, callbacks=callback,
                dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_path', default="./config/mae-vit-base-p16.yaml",
        help='YAML config files')
    parser.add_argument(
        '--use_parallel', default=False, type=str2bool, help='use parallel config.')

    args_ = parse_with_config(parser)

    main(args_)
