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

from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed

from model_utils.config import config
from src.squeezenet import SqueezeNet as squeezenet
from src.dataset import create_dataset_imagenet as create_dataset
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth

set_seed(1)

if __name__ == '__main__':
    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)
    ckpt_save_dir = config.checkpoint_path

    if config.run_distribute:
        if config.device_target == "Ascend":
            # Ascend target
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(
                device_num=config.device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            init()
            ckpt_save_dir = config.checkpoint_path + "ckpt_" + str(get_rank()) + "/"
        elif config.device_target == "GPU":
            # GPU target
            init()
            context.set_auto_parallel_context(
                device_num=get_group_size(),
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            ckpt_save_dir = config.checkpoint_path + "ckpt_" + str(get_rank()) + "/"
        else:
            raise TypeError(f"target type wrong, when run_distribute is Ture, expect [CPU/GPU],"
                            f" but got {config.device_target}")

    # create dataset
    # train dataloader

    train_path = os.path.join(config.train_url, 'train')
    train_dataset = create_dataset(dataset_path=train_path,
                                   do_train=True,
                                   repeat_num=1,
                                   batch_size=config.batch_size,
                                   run_distribute=config.run_distribute)
    step_size = train_dataset.get_dataset_size()

    # eval dataloader
    test_path = os.path.join(config.train_url, 'test')
    eval_dataset = create_dataset(dataset_path=test_path,
                                  do_train=False,
                                  batch_size=config.batch_size,
                                  run_distribute=config.run_distribute)

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    if config.pre_trained and os.path.isfile(config.pre_trained):
        param_dict = load_checkpoint(config.pre_trained)
        for param_name in list(param_dict.keys()):
            if "final_conv." in param_name:
                param_dict.pop(param_name)
        load_param_into_net(net, param_dict)

    # freeze all parameters outside the last
    for param in net.get_parameters():
        if param.name not in ["final_conv.weight", "final_conv.bias"]:
            param.requires_grad = False

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                total_epochs=config.epoch_size,
                warmup_epochs=config.warmup_epochs,
                pretrain_epochs=config.pretrain_epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True,
                              reduction='mean',
                              smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)

    # define opt, model
    if config.device_target == "Ascend":
        # Ascend target
        loss_scale = FixedLossScaleManager(config.loss_scale,
                                           drop_overflow_update=False)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       config.loss_scale,
                       use_nesterov=True)
        model = Model(net,
                      loss_fn=loss,
                      optimizer=opt,
                      loss_scale_manager=loss_scale,
                      metrics={'acc'},
                      amp_level="O2",
                      keep_batchnorm_fp32=False)
    elif config.device_target == "GPU":
        # GPU target
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       use_nesterov=True)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
    elif config.device_target == "CPU":
        # CPU target
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       use_nesterov=True)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc', 'top_1_accuracy', 'top_5_accuracy'})
    else:
        raise TypeError(f"target type wrong, expect [CPU/GPU/Ascend], but got {config.device_target}")

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix='squeezenet_finetune_flowerset',
                                  directory=ckpt_save_dir,
                                  config=config_ck)
        cb += [ckpt_cb]

    model.fit(config.epoch_size - config.pretrain_epoch_size,
              train_dataset, eval_dataset, 10,
              callbacks=cb)
