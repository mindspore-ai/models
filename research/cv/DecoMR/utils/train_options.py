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
import json
import argparse
from collections import namedtuple
import numpy as np


class TrainOptions:
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', default='sample_dp', help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--device_id', type=int, default=1, help='device id')
        gen.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default: false.")
        gen.add_argument('--pretrained', default=True, action='store_true')
        gen.add_argument('--time_to_run', type=int, default=np.inf,
                         help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        gen.add_argument('--ngpu', type=int, default=1, help='Number of gpus used for training')
        gen.add_argument('--rank', type=int, default=0, help='shard_id')
        gen.add_argument('--group_size', type=int, default=1, help='group size')

        io = self.parser.add_argument_group('io')
        io.add_argument('--ckpt_dir', default='./ckpt', help='Directory to store ckp')
        io.add_argument('--eval_dir', default='./ckpt/rank0', help='Directory to store ckpt')
        io.add_argument('--save_root', type=str, default='./results')
        io.add_argument('--log_dir', default='./logs', help='Directory to store logs')
        io.add_argument('--log_freq', default=20, type=int, help='Frequency of printing intermediate results')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None,
                        help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default='/logs/sample_dp/checkpoints/final.pt',
                        help='Load a pretrained network when starting training')

        arch = self.parser.add_argument_group('Architecture')
        arch.add_argument('--model', default='DecoMR', choices=['DecoMR'])

        arch.add_argument('--img_res', type=int, default=224,
                          help='Rescale bounding boxes to size [img_res, img_res] before feeding it in the network')
        arch.add_argument('--uv_res', type=int, default=128, choices=[128, 256],
                          help='The resolution of output location map')
        arch.add_argument('--uv_type', default='BF', choices=['SMPL', 'BF'],
                          help='The type of uv texture map, '
                               'SMPL for SMPL default uv map, '
                               'BF(boundry-free) for our new UV map')

        arch.add_argument('--uv_channels', type=int, default=128, help='Number of channels in uv_map')
        arch.add_argument('--warp_level', type=int, default=2, help='The level of the feature warping process.')
        arch.add_argument('--norm_type', default='GN', choices=['GN', 'BN'],
                          help='Normalization layer of the LNet')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--dataset', default='up-3d',
                           choices=['itw', 'all', 'h36m', 'up-3d', 'mesh', 'spin', 'surreal'],
                           help='Choose training dataset')

        train.add_argument('--num_epochs_dp', type=int, default=5, help='Total number of training epochs in stage dp')
        train.add_argument('--num_epochs_end', type=int, default=30,
                           help='Total number of training epochs in stage end')
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=5000, help='Checkpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')
        train.add_argument('--rot_factor', type=float, default=30,
                           help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument('--noise_factor', type=float, default=0.4,
                           help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument('--scale_factor', type=float, default=0.25,
                           help='rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]')
        train.add_argument('--no_augmentation', dest='use_augmentation', default=True, action='store_false',
                           help='Don\'t do augmentation')
        train.add_argument('--no_augmentation_rgb', dest='use_augmentation_rgb', default=True, action='store_false',
                           help='Don\'t do color jittering during training')
        train.add_argument('--no_flip', dest='use_flip', default=True, action='store_false', help='Don\'t flip images')
        train.add_argument('--stage', default='dp', choices=['dp', 'end'],
                           help='Training stage, '
                                'dp: only train the CNet'
                                'end: end-to-end training.')

        train.add_argument('--use_spin_fit', dest='use_spin_fit', default=False, action='store_true',
                           help='Use the fitting result from spin as GT')
        train.add_argument('--adaptive_weight', dest='adaptive_weight', default=False, action='store_true',
                           help='Change the loss weight according to the fitting error of SPIN fit results.'
                                'Useful only if use_spin_fit = True.')
        train.add_argument('--gtkey3d_from_mesh', dest='gtkey3d_from_mesh', default=False, action='store_true',
                           help='For the data without GT 3D keypoints but with fitted SMPL parameters,'
                                'get the GT 3D keypoints from the mesh.')

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)

        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
        optim.add_argument("--wd", type=float, default=0, help="Weight decay weight")
        optim.add_argument("--lam_tv", type=float, default=1e-4, help='lambda of tv loss')
        optim.add_argument("--lam_con", type=float, default=1, help='lambda of consistent loss')
        optim.add_argument("--lam_dp_mask", type=float, default=0.2, help='lambda of densepose mask loss')
        optim.add_argument("--lam_dp_uv", type=float, default=1, help='lambda of densepose uv loss')
        optim.add_argument("--lam_mesh", type=float, default=0, help='lambda of mesh loss')
        optim.add_argument("--lam_uv", type=float, default=1, help='lambda of location map loss')
        optim.add_argument("--lam_key2d", type=float, default=1, help='lambda of 2D joint loss')
        optim.add_argument("--lam_key3d", type=float, default=1, help='lambda of 3D joint loss')

        train.add_argument('--use_smpl_joints', dest='use_smpl_joints', default=False, action='store_true',
                           help='Use the 24 SMPL joints for supervision, '
                                'should be set True when using data from SURREAL dataset.')
        optim.add_argument("--lam_key2d_smpl", type=float, default=1, help='lambda of 2D SMPL joint loss')
        optim.add_argument("--lam_key3d_smpl", type=float, default=1, help='lambda of 3D SMPL joint loss')


    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)

            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)

            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
