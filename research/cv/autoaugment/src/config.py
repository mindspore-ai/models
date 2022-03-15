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
Configurable parameters.
"""

import argparse
import logging

logger = logging.getLogger('config')


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

        # The dataset to train or test against. Choices: ['cifar10', 'svhn']
        self.dataset = 'cifar10'
        # The number of classes.
        self.class_num = 10
        # Path to the folder where the intended dataset is stored.
        self.dataset_path = './cifar-10-batches-bin'
        # Batch size for both training mode and testing mode.
        self.batch_size = 128
        # Indicates training or testing mode.
        self.training = training

        self.dataset_sink_mode = False

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

            if self.run_distribute is True:
                self.device_num = None
            # The pre-trained checkpoint to load and train from.
            # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
            self.pre_trained = None

            # Number of epochs to train.
            self.epoch_size = 200  # 50 for svhn dataset
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
            self.save_checkpoint_epochs = 5
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

    def hook_device_target(self, x):
        """Sets the device_target value."""
        self.device_target = x

    def hook_dataset(self, x):
        """Sets the dataset value."""
        self.dataset = x

    def hook_dataset_path(self, x):
        """Sets the dataset_path value."""
        self.dataset_path = x

    def hook_checkpoint_path(self, x):
        """Sets the checkpoint_path value."""
        self.checkpoint_path = x

    def hook_augment(self, x):
        """Sets the augment value."""
        self.augment = x

    def hook_device_num(self, x):
        """Sets the device_num value."""
        self.device_num = x

    def hook_distribute(self, x):
        """Sets the run_distribute value."""
        self.run_distribute = x

    def hook_lr_max(self, x):
        """Sets the lr_max value."""
        self.lr_max = x

    def hook_lr_init(self, x):
        """Sets the lr_init value."""
        self.lr_init = x
    def hook_epoch_size(self, x):
        """Sets the epoch_size value."""
        self.epoch_size = x

    def hook_pre_trained(self, x):
        """Sets the pre_trained value."""
        self.pre_trained = x

    def hook_save_checkpoint_path(self, x):
        """Sets the save_checkpoint_path value."""
        self.save_checkpoint_path = x

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
        hooks['device_target'] = self.hook_device_target

        parser.add_argument(
            '--dataset', type=str, default='cifar10',
            choices=['cifar10', 'svhn'],
            help='The dataset to train or test against.',
        )
        hooks['dataset'] = self.hook_dataset

        parser.add_argument(
            '--dataset_path', type=str, default='./cifar-10-batches-bin',
            help='Path to the folder where the intended dataset is stored.',
        )
        hooks['dataset_path'] = self.hook_dataset_path

        if not self.training:
            parser.add_argument(
                '--checkpoint_path', type=str, default=None,
                help='The checkpoint to load and test against. '
                     'Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt',
            )
            hooks['checkpoint_path'] = self.hook_checkpoint_path

        if self.training:
            parser.add_argument(
                '--augment', type=bool, default=True,
                help='Whether to apply auto-augment or not.',
            )
            hooks['augment'] = self.hook_augment

            parser.add_argument(
                '--device_num', type=int, default=1,
                help='The number of device(s) to be used for training.',
            )
            hooks['device_num'] = self.hook_device_num

            parser.add_argument(
                '--run_distribute', type=bool, default=False,
                help='Whether to train the model in distributed mode or not.',
            )
            hooks['run_distribute'] = self.hook_distribute

            parser.add_argument(
                '--lr_max', type=float, default=0.1,
                help='The maximum learning rate.',
            )
            hooks['lr_max'] = self.hook_lr_max

            parser.add_argument(
                '--lr_init', type=float, default=0.1,
                help='The init learning rate.',
            )
            hooks['lr_init'] = self.hook_lr_init
            parser.add_argument(
                '--epoch_size', type=int, default=200,
                help='Number of epochs.',
            )
            hooks['epoch_size'] = self.hook_epoch_size

            parser.add_argument(
                '--pre_trained', type=str, default=None,
                help='The pre-trained checkpoint to load and train from. '
                     'Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt',
            )
            hooks['pre_trained'] = self.hook_pre_trained

            parser.add_argument(
                '--save_checkpoint_path', type=str, default='./checkpoint',
                help='The folder path to save checkpoints.',
            )
            hooks['save_checkpoint_path'] = self.hook_save_checkpoint_path

        # Overwrite default configurations by cli arguments
        args_opt = parser.parse_args()
        for name, val in args_opt.__dict__.items():
            hooks[name](val)
