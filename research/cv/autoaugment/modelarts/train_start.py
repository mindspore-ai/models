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
"""
Model training entrypoint.
"""

import os
import argparse
import logging
import glob
import ast
import numpy as np
from mindspore import context, Model, load_checkpoint, load_param_into_net, Tensor, export
from mindspore.common import set_seed
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, \
    LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import moxing as mox
from src.dataset import create_cifar10_dataset
from src.network import WRN
from src.optim import get_lr
from src.utils import init_utils


def model_export(arguments):
    """export air"""
    output_dir = arguments.local_output_dir
    ckpt_file = glob.glob(output_dir + '/' + '*.ckpt')[0]
    network = WRN(160, 3, conf.class_num)
    print("ckpt_file: ", ckpt_file)
    param_dic = load_checkpoint(ckpt_file)
    load_param_into_net(net, param_dic)
    image = Tensor(np.ones((1, 3, 32, 32), np.float32))
    export_file = os.path.join(output_dir, conf.file_name)
    export(network, image, file_name=export_file, file_format=conf.file_format)
    return 0

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

class Config:
    """
    Define configurable parameters.

    Args:
        training (bool): Whether it is used in training mode or testing mode.
        load_args (bool): Whether to load cli arguments automatically or not.
    """

    def __init__(self, training, load_args=True):
        # Set to mute logs with lower levels.
        self.log_level = logging.INFO

        # Random seed.
        self.seed = 1

        # Type of device(s) where the model would be deployed to.
        # Choices: ['Ascend', 'GPU', 'CPU']
        self.device_target = 'Ascend'

        # The model to use. Choices: ['wrn']
        self.net = 'wrn'

        # The dataset to train or test against. Choices: ['cifar10']
        self.dataset = 'cifar10'
        # The number of classes.
        self.class_num = 10
        # Path to the folder where the intended dataset is stored.
        self.dataset_path = '../dataset/cifar-10-batches-bin'

        # Batch size for both training mode and testing mode.
        self.batch_size = 128

        # Indicates training or testing mode.
        self.training = training

        # Testing parameters.
        if not self.training:
            # The checkpoint to load and test against.
            # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
            self.checkpoint_path = None

        # Training parameters.
        if self.training:
            # Whether to apply auto-augment or not.
            self.augment = True

            # The number of device(s) to be used for training.
            self.device_num = 1
            # Whether to train the model in a distributed mode or not.
            self.run_distribute = False
            # The pre-trained checkpoint to load and train from.
            # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
            self.pre_trained = False

            # Number of epochs to train.
            self.epoch_size = 10
            # Momentum factor.
            self.momentum = 0.9
            # L2 penalty.
            self.weight_decay = 5e-4
            # Learning rate decaying mode. Choices: ['cosine']
            self.lr_decay_mode = 'cosine'
            # The starting learning rate.
            self.lr_init = 0.1
            # The maximum learning rate.
            self.lr_max = 0.1
            # The number of warmup epochs. Note that during the warmup period,
            # the learning rate grows from `lr_init` to `lr_max` linearly.
            self.warmup_epochs = 5
            # Loss scaling for mixed-precision training.
            self.loss_scale = 1024

            # Create a checkpoint per `save_checkpoint_epochs` epochs.
            self.save_checkpoint_epochs = 10
            # The maximum number of checkpoints to keep.
            self.keep_checkpoint_max = 10
            # The folder path to save checkpoints.
            self.save_checkpoint_path = './checkpoint'

        # _init is an initialization guard, which helps warn setting attributes
        # outside __init__.
        self._init = True
        if load_args:
            self.load_args()

    def __setattr__(self, name, value):
        """___setattr__ is customized to warn adding attributes outside
        __init__ and encourage declaring configurable parameters explicitly in
        __init__."""
        if getattr(self, '_init', False) and not hasattr(self, name):
            logger.warning('attempting to add an attribute '
                           'outside __init__: %s=%s', name, value)
        object.__setattr__(self, name, value)

    def load_args(self):
        """load_args overwrites configurations by cli arguments."""
        hooks = {}  # hooks are used to assign values.
        parser = argparse.ArgumentParser(
            description='AutoAugment for image classification.')

        parser.add_argument(
            '--device_target', type=str, default='Ascend',
            choices=['Ascend', 'GPU', 'CPU'],
            help='Type of device(s) where the model would be deployed to.',
        )
        def hook_device_target(x):
            """Sets the device_target value."""
            self.device_target = x
        hooks['device_target'] = hook_device_target

        parser.add_argument(
            '--dataset', type=str, default='cifar10',
            choices=['cifar10'],
            help='The dataset to train or test against.',
        )
        def hook_dataset(x):
            """Sets the dataset value."""
            self.dataset = x
        hooks['dataset'] = hook_dataset

        parser.add_argument(
            '--dataset_path', type=str, default='../dataset/cifar-10-batches-bin',
            help='Path to the folder where the intended dataset is stored.',
        )
        def hook_dataset_path(x):
            """Sets the dataset_path value."""
            self.dataset_path = x
        hooks['dataset_path'] = hook_dataset_path

        if not self.training:
            parser.add_argument(
                '--checkpoint_path', type=str, default=None,
                help='The checkpoint to load and test against. '
                     'Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt',
            )
            def hook_checkpoint_path(x):
                """Sets the checkpoint_path value."""
                self.checkpoint_path = x
            hooks['checkpoint_path'] = hook_checkpoint_path

        if self.training:
            parser.add_argument(
                '--augment', type=ast.literal_eval, default=True,
                help='Whether to apply auto-augment or not.',
            )
            def hook_augment(x):
                """Sets the augment value."""
                self.augment = x
            hooks['augment'] = hook_augment

            parser.add_argument(
                '--device_num', type=int, default=1,
                help='The number of device(s) to be used for training.',
            )
            def hook_device_num(x):
                """Sets the device_num value."""
                self.device_num = x
            hooks['device_num'] = hook_device_num

            parser.add_argument(
                '--run_distribute', type=ast.literal_eval, default=False,
                help='Whether to train the model in distributed mode or not.',
            )
            def hook_distribute(x):
                """Sets the run_distribute value."""
                self.run_distribute = x
            hooks['run_distribute'] = hook_distribute

            parser.add_argument(
                '--lr_max', type=float, default=0.1,
                help='The maximum learning rate.',
            )
            def hook_lr_max(x):
                """Sets the lr_max value."""
                self.lr_max = x
            hooks['lr_max'] = hook_lr_max

            parser.add_argument(
                '--pre_trained', type=bool, default=False,
                help='The pre-trained checkpoint to load and train from. '
                     'Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt',
            )
            def hook_pre_trained(x):
                """Sets the pre_trained value."""
                self.pre_trained = x
            hooks['pre_trained'] = hook_pre_trained

            parser.add_argument(
                '--save_checkpoint_path', type=str, default='./checkpoint',
                help='The folder path to save checkpoints.',
            )
            def hook_save_checkpoint_path(x):
                """Sets the save_checkpoint_path value."""
                self.save_checkpoint_path = x
            hooks['save_checkpoint_path'] = hook_save_checkpoint_path

            parser.add_argument('--train_url', required=True, default=None, help='obs browser path')
            def hook_train_url(x):
                """Sets the save_checkpoint_path value."""
                self.train_url = x
            hooks['train_url'] = hook_train_url

            parser.add_argument('--data_url', required=True, default=None, help='Location of data')
            def hook_data_url(x):
                """Sets the save_checkpoint_path value."""
                self.data_url = x
            hooks['data_url'] = hook_data_url

            parser.add_argument('--local_data_dir', type=str, default="/cache")
            def local_data_dir(x):
                """Sets the save_checkpoint_path value."""
                self.local_data_dir = x
            hooks['local_data_dir'] = local_data_dir

            parser.add_argument('--local_output_dir', type=str, default="/cache/train_output")
            def local_output_dir(x):
                """Sets the save_checkpoint_path value."""
                self.local_output_dir = x
            hooks['local_output_dir'] = local_output_dir

            parser.add_argument('--file_format', type=str, choices=['AIR', 'MINDIR'], default='AIR')
            def file_format(x):
                """Sets the save_checkpoint_path value."""
                self.file_format = x
            hooks['file_format'] = file_format

            parser.add_argument('--file_name', type=str, default="wrn-autoaugment")
            def file_name(x):
                """Sets the save_checkpoint_path value."""
                self.file_name = x
            hooks['file_name'] = file_name

            parser.add_argument('--class_num', type=int, default=10, help='class_num')
            def class_num(x):
                """Sets the save_checkpoint_path value."""
                self.class_num = x
            hooks['class_num'] = class_num

            parser.add_argument('--epoch_size', type=int, default=180, help='epoch_size')
            def epoch_size(x):
                """Sets the save_checkpoint_path value."""
                self.epoch_size = x
            hooks['epoch_size'] = epoch_size

        # Overwrite default configurations by cli arguments
        args_opt = parser.parse_args()
        for name, val in args_opt.__dict__.items():
            hooks[name](val)


if __name__ == '__main__':

    logger = logging.getLogger('config')
    conf = Config(training=True)
    init_utils(conf)
    set_seed(conf.seed)

    local_data_path = conf.local_data_dir
    train_output_path = conf.local_output_dir
    mox.file.copy_parallel(conf.data_url, local_data_path)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~file copy success~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # Initialize context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=conf.device_target,
        save_graphs=False,
    )
    if conf.run_distribute:
        if conf.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(
                device_id=device_id
            )
            context.set_auto_parallel_context(
                device_num=conf.device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
            init()
        elif conf.device_target == 'GPU':
            init()
            context.set_auto_parallel_context(
                device_num=get_group_size(),
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
    else:
        try:
            device_id = int(os.getenv('DEVICE_ID'))
        except TypeError:
            device_id = 0
        context.set_context(device_id=device_id)

    # Create dataset
    if conf.dataset == 'cifar10':
        dataset = create_cifar10_dataset(
            dataset_path=conf.data_url,
            do_train=True,
            repeat_num=1,
            batch_size=conf.batch_size,
            target=conf.device_target,
            distribute=conf.run_distribute,
            augment=conf.augment,
        )
    step_size = dataset.get_dataset_size()

    # Define net
    net = WRN(160, 3, conf.class_num)

    # Initialize learning rate
    lr = Tensor(get_lr(
        lr_init=conf.lr_init, lr_max=conf.lr_max,
        warmup_epochs=conf.warmup_epochs, total_epochs=conf.epoch_size,
        steps_per_epoch=step_size, lr_decay_mode=conf.lr_decay_mode,
    ))

    # Define loss, opt, and model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(
        conf.loss_scale,
        drop_overflow_update=False,
    )
    opt = Momentum(
        filter(lambda x: x.requires_grad, net.get_parameters()),
        lr, conf.momentum, conf.weight_decay, conf.loss_scale,
    )
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, metrics={'acc'})

    # Define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(
        save_checkpoint_steps=conf.save_checkpoint_epochs * step_size,
        keep_checkpoint_max=conf.keep_checkpoint_max,
    )
    ck_cb = ModelCheckpoint(
        prefix='train_%s_%s' % (conf.net, conf.dataset),
        directory=conf.save_checkpoint_path,
        config=config_ck,
    )
    if not os.path.exists(conf.local_output_dir):
        os.mkdir(conf.local_output_dir)

    # Train
    if conf.run_distribute:
        callbacks = [time_cb, loss_cb]
        if conf.device_target == 'GPU' and str(get_rank()) == '0':
            callbacks = [time_cb, loss_cb, ck_cb]
        elif conf.device_target == 'Ascend' and device_id == 0:
            callbacks = [time_cb, loss_cb, ck_cb]
    else:
        callbacks = [time_cb, loss_cb, ck_cb]

    model.train(conf.epoch_size, dataset, callbacks=callbacks)
    conf.local_output_dir = conf.save_checkpoint_path
    model_export(conf)
    mox.file.copy_parallel(conf.local_output_dir, conf.train_url)
