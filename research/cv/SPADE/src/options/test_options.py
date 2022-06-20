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
""" test options """

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--fid_eval_ckpt_dir', type=str, \
                            default='./inception/inception_pid.ckpt', help='saves results here.')

        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--ckpt_dir', type=str, \
                            default='./checkpoints/netG_epoch_200.ckpt', help='saves results here.')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', \
                            crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
