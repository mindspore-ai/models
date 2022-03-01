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

import glob
import os

import moxing as mox

import train as model_train
from export import run_export
from src.model_utils.config import config


def get_last_ckpt():
    ckpt_pattern = os.path.join(config.save_checkpoint_path, '0/*.ckpt')
    ckpt_list = glob.glob(ckpt_pattern)
    if not ckpt_list:
        print(f"Cant't found ckpt in {config.save_checkpoint_path}")
        return None
    ckpt_list.sort(key=os.path.getmtime)
    print("====================%s" % ckpt_list[-1])
    return ckpt_list[-1]


def export_air():
    config.ckpt_file = get_last_ckpt()
    if not config.ckpt_file:
        return
    config.file_name = os.path.join(config.output_path, config.file_name)
    config.enable_modelarts = False
    run_export()


def main():
    model_train.train()
    export_air()
    mox.file.copy_parallel(config.output_path, config.train_url)


if __name__ == '__main__':
    main()
    