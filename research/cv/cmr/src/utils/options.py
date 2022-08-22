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
import argparse


class TrainOptions:
    """Object that handles command line options for training."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options to train the model")

        config = self.parser.add_argument_group("Config")
        config.add_argument('--save_checkpoint_dir', default=None, help="Path to save checkpoint")
        config.add_argument('--keep_checkpoint_max', type=int, default=10, help="Maximum number of checkpoints")
        config.add_argument('--pretrained_checkpoint', default=None,
                            help="Load a pretrained Graph CNN when starting training")
        config.add_argument('--device_target', default='GPU', choices=['GPU', 'Ascend'],
                            help="Type of device to perform experiment")

        arch = self.parser.add_argument_group("Architecture")
        arch.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers')
        arch.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN')
        arch.add_argument('--img_res', type=int, default=224,
                          help='Rescale bounding boxes to size [img_res, img_res] before feeding it in the network')

        train = self.parser.add_argument_group("Training options")
        train.add_argument('--num_epochs', type=int, default=50, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')
        train.add_argument('--rot_factor', type=float, default=30,
                           help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument('--noise_factor', type=float, default=0.4,
                           help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument('--scale_factor', type=float, default=0.25,
                           help='rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]')
        train.add_argument('--do_shuffle', dest='do_shuffle', default=False, action='store_true',
                           help="Shuffle training dataset")
        train.add_argument('--distribute', dest='distribute', default=False, action='store_true',
                           help='Distributed training')
        train.add_argument('--num_workers', type=int, default=2, help="Number of worker to generate dataset")

        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
        optim.add_argument("--wd", type=float, default=0, help="Weight decay weight")

    def parse_args(self):
        """Parse input arguments."""
        args = self.parser.parse_args()
        try:
            if not os.path.exists(args.save_checkpoint_dir):
                os.mkdir(args.save_checkpoint_dir)
        except OSError as e:
            print(e)
        return args


class EvalOptions:
    """Object that handles command line options for evaluation."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options to eval the model")

        self.parser.add_argument('--num_channels', type=int, default=256,
                                 help='Number of channels in Graph Residual layers')
        self.parser.add_argument('--num_layers', type=int, default=5,
                                 help='Number of residuals blocks in the Graph CNN')
        self.parser.add_argument('--img_res', type=int, default=224,
                                 help='Rescale bounding boxes to size [img_res, img_res] '
                                      'before feeding it in the network')
        self.parser.add_argument('--checkpoint', default=None, required=True, help='Path to network checkpoint')
        self.parser.add_argument('--dataset', default='up-3d', choices=['up-3d'],
                                 help='Choose evaluation dataset')
        self.parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
        self.parser.add_argument('--shuffle', default=False, action='store_true', help="Shuffle data")
        self.parser.add_argument('--num_workers', default=2, type=int, help="Number of processes for data loading")
        self.parser.add_argument('--log_freq', default=50, type=int, help="Frequency of printing intermediate results")
        self.parser.add_argument('--device_target', default='GPU', type=str, help="Type of device to eval model")

    def parse_args(self):
        """Parse input arguments."""
        args = self.parser.parse_args()
        if not os.path.isfile(args.checkpoint):
            raise ValueError("Checkpoint is not exist")
        return args
