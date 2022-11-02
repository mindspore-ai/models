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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" train options """

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--vgg_ckpt_path', type=str, default='')
        # for training
        parser.add_argument('--which_epoch', type=str, default='latest', \
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--decay_epoch', type=int, default=100)
        parser.add_argument('--total_epoch', type=int, default=200)
        parser.add_argument('--now_epoch', type=int, default=0)
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')

        parser.add_argument('--G_lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--D_lr', type=float, default=0.0004, help='initial learning rate for adam')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        self.isTrain = True
        return parser
