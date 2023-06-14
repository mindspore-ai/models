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
"""transfer training."""
import os
import datetime
import mindspore as ms

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.common import set_seed
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from src.dataset import classification_dataset
from src.crossentropy import CrossEntropy
from src.lr_generator import get_lr
from src.utils.logging import get_logger
from src.utils.optimizers__init__ import get_param_groups
from src.image_classification import get_network
from src.model_utils.config import config


set_seed(1)


def set_parameters():
    """parameters"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    # init distributed
    if config.run_distribute:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1

    if config.is_dynamic_loss_scale == 1:
        config.loss_scale = 1  # for dynamic loss scale can not set loss scale in momentum opt

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.output_path, datetime.datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S"))
    config.logger = get_logger(config.outputs_dir, config.rank)
    return config


def load_pretrain_model(ckpt_file, network, args):
    """load pretrain model."""
    if os.path.isfile(ckpt_file):
        param_dict = load_checkpoint(ckpt_file)
        for param_name in list(param_dict.keys()):
            if ".fc." in param_name:
                param_dict.pop(param_name)
        load_param_into_net(network, param_dict)
        args.logger.info("load model {} success".format(ckpt_file))


def eval_net(net, dataset):
    """eval net"""
    net.set_train(False)
    from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={"top_1_accuracy", "top_5_accuracy"})

    # eval model
    res = model.eval(dataset)
    print("result:", res)


def fine_train():
    """training process"""
    set_parameters()
    context.set_context(device_id=config.device_id)

    # train dataloader
    train_path = os.path.join(config.data_path, "train")
    de_dataset = classification_dataset(
        train_path, config.image_size, config.per_batch_size, 1, config.rank, config.group_size, num_parallel_workers=4
    )

    config.steps_per_epoch = de_dataset.get_dataset_size()

    # eval dataloader
    test_path = os.path.join(config.data_path, "test")
    val_dataset = classification_dataset(
        test_path,
        image_size=config.image_size,
        per_batch_size=config.per_batch_size,
        max_epoch=1,
        rank=config.rank,
        group_size=config.group_size,
        mode="eval",
    )
    config.logger.save_args(config)
    # network
    config.logger.important_info("start create network")

    # get network and init
    network = get_network(network=config.network, num_classes=config.num_classes, platform=config.device_target)

    # read ckpt
    load_pretrain_model(config.checkpoint_file_path, network, config)

    # freeze all parameters outside the last
    for param in network.get_parameters():
        if param.name not in ["head.fc.weight", "head.fc.bias", "moments.head.fc.weight", "moments.head.fc.bias"]:
            param.requires_grad = False

    # lr scheduler
    lr = get_lr(config)

    # optimizer
    opt = Momentum(
        params=get_param_groups(network),
        learning_rate=Tensor(lr),
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        loss_scale=config.loss_scale,
    )

    # loss
    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(
        network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, metrics={"acc"}, amp_level="O3"
    )

    # checkpoint save
    callbacks = [TimeMonitor(data_size=config.steps_per_epoch), LossMonitor(per_print_times=1)]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=config.ckpt_interval * config.steps_per_epoch,
            keep_checkpoint_max=config.ckpt_save_max,
        )
        save_ckpt_path = os.path.join(config.outputs_dir, "ckpt_" + str(config.rank) + "/")
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=save_ckpt_path, prefix="{}".format(config.rank))
        callbacks.append(ckpt_cb)

    model.train(config.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=False)

    # evaluate
    eval_net(network, val_dataset)


if __name__ == "__main__":
    fine_train()
