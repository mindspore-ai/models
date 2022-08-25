# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
APDrawingGAN testing option.
"""

import os
import argparse
from src.option.config import get_config


class TestOptions():
    """TestOptions"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="default name", add_help=False)
        self.parser.add_argument("--dataroot", type=str, default='test_dataset/data/test_single')
        self.parser.add_argument('--lm_dir', type=str, default='test_dataset/landmark/ALL', \
                                 help='path to facial landmarks')
        self.parser.add_argument('--bg_dir', type=str, default='test_dataset/mask/ALL', help='path to background masks')
        self.parser.add_argument('--results_dir', type=str, default='test_dataset/result', help='path to test results')
        self.parser.add_argument('--model_path', type=str, default='checkpoint/netG_300.ckpt')
        self.parser.add_argument('--onnx_path', type=str, default='onnx_file/')
        self.parser.add_argument('--mindir_filename', type=str, default='infer_model')
        self.parser.add_argument('--onnx_filename', type=str, default='apdrawinggan_onnx')
        self.parser.add_argument('--config_path', type=str, default="config_eval_and_export.yaml")
        self.parser.add_argument('--device_id', type=int, default=0, help='device id')
        self.parser.add_argument("--device_target", type=str, default="Ascend", help="GPU or Ascned")
        self.parser.add_argument('--isExport', type=bool, default=False, help='modelarts')

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
            self.opt.dataroot = os.path.join(self.settings.modelarts_data_dir, "data/test_single")
            self.opt.lm_dir = os.path.join(self.settings.modelarts_data_dir, "landmark/ALL")
            self.opt.bg_dir = os.path.join(self.settings.modelarts_data_dir, "mask/ALL")
            self.opt.model_path = os.path.join(self.settings.modelarts_data_dir, self.settings.model_path)
            self.opt.results_dir = self.settings.modelarts_result_dir
            self.opt.mindir_filename = os.path.join(self.settings.modelarts_result_dir, self.settings.mindir_filename)
        else:
            self.opt = get_config(self.parser, self.settings.config_path)

    def get_settings(self):
        return self.opt
