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
"""
APDrawingGAN train option.
"""

import os
import argparse
from src.option.config import get_config

class Options():
    """Options"""
    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description="default name", add_help=False)
        self.parser.add_argument("--device_target", type=str, default="Ascend", help="GPU or Ascned")
        self.parser.add_argument("--dataroot", type=str, default='')
        self.parser.add_argument("--ckpt_dir", type=str, default='checkpoint')
        self.parser.add_argument("--auxiliary_dir", type=str, default='/data/model_weight/')
        self.parser.add_argument("--mindrecord_dir", type=str, default='dataset/training_dataset.mindrecord')
        self.parser.add_argument('--lm_dir', type=str, default='dataset/landmark/ALL', help='path to facial landmarks')
        self.parser.add_argument('--bg_dir', type=str, default='dataset/mask/ALL', help='path to background masks')
        self.parser.add_argument('--device_id', type=int, default=0, help='device id')
        self.parser.add_argument('--group_size', type=int, default=1, help='group size')
        self.parser.add_argument('--rank', type=int, default=0, help='rank id')
        self.parser.add_argument('--use_local', action='store_true', help='use local part network')
        self.parser.add_argument('--no_flip', action='store_false')
        self.parser.add_argument('--save_epoch_freq', type=int, default=25,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate')
        self.parser.add_argument('--isExport', type=bool, default=False, help='modelarts')

        # =====================================auxiliary net structure===============================================
        # multiple discriminators
        self.parser.add_argument('--discriminator_local', action='store_true',
                                 help='use six diffent local discriminator for 6 local regions')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--pretrain', action='store_true', help='train')
        self.parser.add_argument('--isTrain', action='store_true', help='train')

        # ========================================distribute=========================================================
        self.parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default: false.")
        self.parser.add_argument('--config_path', type=str, default="config_train.yaml")

        # ========================================modelarts=========================================================
        self.parser.add_argument('--isModelarts', type=bool, default=False, help='modelarts')
        self.parser.add_argument("--train_url", type=str, default="./output")
        self.parser.add_argument("--data_url", type=str, default="./data")
        self.parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
        self.parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
        self.settings = self.parser.parse_known_args()[0]
        if self.settings.isModelarts:
            from src.utils.tools import obs_data2modelarts
            obs_data2modelarts(self.settings)
            self.settings.config_path = os.path.join(self.settings.modelarts_data_dir, self.settings.config_path)
            self.opt = get_config(self.parser, self.settings.config_path)
            self.opt.dataroot = os.path.join(self.settings.modelarts_data_dir, "data/train")
            self.opt.auxiliary_dir = os.path.join(self.settings.modelarts_data_dir, "auxiliary.ckpt")
            self.opt.lm_dir = os.path.join(self.settings.modelarts_data_dir, "landmark/ALL")
            self.opt.bg_dir = os.path.join(self.settings.modelarts_data_dir, "mask/ALL")
            self.opt.use_local = True
            self.opt.discriminator_local = True
            self.opt.no_flip = False
            self.opt.no_dropout = True
            self.opt.pretrain = True
            self.opt.isTrain = True
            self.opt.ckpt_dir = os.path.join(self.settings.modelarts_result_dir, self.settings.ckpt_dir)
            print(self.opt)
        else:
            self.opt = get_config(self.parser, self.settings.config_path)

    def get_settings(self):
        return self.opt
