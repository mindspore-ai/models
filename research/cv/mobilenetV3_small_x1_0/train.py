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
"""train_imagenet."""

import os
from mindspore import context
from mindspore import Tensor

from mindspore.nn import RMSProp
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size

from src.dataset import create_dataset
from src.lr_generator import get_lr
from src.config import config_ascend, config_gpu
from src.loss import CrossEntropyWithLabelSmooth
from src.monitor import Monitor
from src.mobilenetv3 import mobilenet_v3_small
from argparser import arg_parser


def get_loss(label_smooth, num_classes):
    if label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=label_smooth, num_classes=num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss

def main(args_opt):
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)
    device_num = int(os.getenv('RANK_SIZE', '1'))

    config = config_ascend if args_opt.device_target == 'Ascend' else config_gpu

    # init distributed
    if args_opt.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel', gradients_mean=True)
            local_data_url = os.path.join(local_data_url, str(device_id))
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        # define dataset
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    else:
        if args_opt.run_distribute:
            if args_opt.device_target == 'Ascend':
                device_id = int(os.getenv('DEVICE_ID'))
                context.set_context(device_id=device_id)
                init()
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num,
                                                  parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
            elif args_opt.device_target == "GPU":
                init()
                device_id = get_rank()
                device_num = get_group_size()
                context.set_auto_parallel_context(device_num=device_num,
                                                  parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
        else:
            device_id = int(args_opt.device_id)
            context.set_context(device_id=device_id)
            device_num = 1
        # define dataset
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    step_size = dataset.get_dataset_size()
    # define net
    net = mobilenet_v3_small(num_classes=config.num_classes, multiplier=1.)
    # define loss
    loss = get_loss(config.label_smooth, config.num_classes)

    # resume
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    # define optimizer
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(global_step=0,
                       lr_init=0,
                       lr_end=0,
                       lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size))
    opt = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                  momentum=config.momentum, epsilon=0.001, loss_scale=config.loss_scale)
    # define model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, amp_level='O3')

    cb = [Monitor(lr_init=lr.asnumpy())]

    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = None
        if device_num == 1 or device_id == 0:
            save_ckpt_path = local_train_url if args_opt.run_modelarts else \
                os.path.join(config.save_checkpoint_path, 'model_' + str(device_id) + '/')
        if args_opt.device_target == "GPU" and args_opt.run_distribute:
            save_ckpt_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(device_id) + "/")

        if save_ckpt_path is not None:
            ckpt_cb = ModelCheckpoint(prefix="mobilenetV3", directory=save_ckpt_path, config=config_ck)
            cb += [ckpt_cb]
    # begin train
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    if args_opt.run_modelarts and config.save_checkpoint and (device_num == 1 or device_id == 0):
        mox.file.copy_parallel(local_train_url, args_opt.train_url)


if __name__ == '__main__':
    main(arg_parser())
