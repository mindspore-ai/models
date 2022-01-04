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

import argparse
import os
import yaml


class Options():
    def __init__(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path, "../src/config.yaml")
        file = open(yaml_path, 'r', encoding='utf-8')
        cfg_yaml = file.read()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--device_target', type=str, default='Ascend')
        self.parser.add_argument('--dataset_path', type=str, default=None)
        self.parser.add_argument('--data_url', type=str, default=None)
        self.parser.add_argument('--train_url', type=str, default=None)
        self.parser.add_argument('--distribute', type=int, default=False)
        self.parser.add_argument('--model_arts', type=int, default=False)
        self.arg = self.parser.parse_args()
        self.opt = yaml.safe_load(cfg_yaml)
        self.opt["input_channel"] = 1 if self.opt["grayscale"] else 3

    def parse(self):
        local_path = '/cache/user-job-dir/ssim-ae/'
        self.opt["dataset_path"] = self.arg.dataset_path
        self.opt["device_target"] = self.arg.device_target
        self.opt["distribute"] = not self.arg.distribute == 0
        self.opt["model_arts"] = not self.arg.model_arts == 0
        if self.opt["model_arts"]:
            self.opt["data_url"] = self.arg.data_url
            self.opt["train_url"] = self.arg.train_url
            import moxing as mox
            dataset_path = local_path + 'data'
            mox.file.copy_parallel(src_url=self.opt["data_url"], dst_url=dataset_path)
        else:
            dataset_path = self.opt["dataset_path"]
        if self.opt.get("train_data_dir", None) is None:
            self.opt["train_data_dir"] = dataset_path + '/train'
        if self.opt.get("test_dir", None) is None:
            self.opt["test_dir"] = dataset_path + '/test'
        if self.opt.get("sub_folder", None) is None:
            self.opt["sub_folder"] = os.listdir(self.opt["test_dir"])
        if self.opt.get("aug_dir", None) is None:
            if self.opt["model_arts"]:
                self.opt["aug_dir"] = local_path + 'train_patches/'
            else:
                if self.opt["distribute"]:
                    self.opt["aug_dir"] = '../train_patches/'
                else:
                    self.opt["aug_dir"] = './train_patches/'
        if self.opt.get("checkpoint_dir", None) is None:
            if self.opt["model_arts"]:
                self.opt["checkpoint_dir"] = local_path + '/results/checkpoints/'
            else:
                if self.opt["distribute"]:
                    self.opt["checkpoint_dir"] = '../results/checkpoints/'
                else:
                    self.opt["checkpoint_dir"] = './results/checkpoints/'
        if os.getenv('RANK_ID', '0') == "0":
            if not os.path.exists(self.opt["checkpoint_dir"]):
                os.makedirs(self.opt["checkpoint_dir"])
            if not os.path.exists(self.opt["aug_dir"]):
                os.makedirs(self.opt["aug_dir"])

        self.opt["input_channel"] = 1 if self.opt["grayscale"] else 3
        self.opt["mask_size"] = self.opt["data_augment"]["crop_size"] \
            if self.opt["data_augment"]["im_resize"] - \
            self.opt["data_augment"]["crop_size"] < \
            self.opt["stride"] else \
            self.opt["data_augment"]["im_resize"]
        return self.opt


class Options_310():
    def __init__(self):
        curPath = os.path.dirname(os.path.realpath(__file__))
        yamlPath = os.path.join(curPath, "../src/config.yaml")
        f = open(yamlPath, 'r', encoding='utf-8')
        cfg_yaml = f.read()
        self.opt = yaml.safe_load(cfg_yaml)
        self.opt["input_channel"] = 1 if self.opt["grayscale"] else 3
        self.opt["mask_size"] = self.opt["data_augment"]["crop_size"] \
            if self.opt["data_augment"]["im_resize"] - \
            self.opt["data_augment"]["crop_size"] < \
            self.opt["stride"] else \
            self.opt["data_augment"]["im_resize"]
