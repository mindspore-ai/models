"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""

import os
import export
import train
import moxing as mox
from model_utils.config import config

def init_argument():
    '''init argument '''
    config.enable_modelarts = True
    config.data_dir = '/cache/data/train_data'
    config.ckpt_path = './output'
    config.need_modelarts_dataset_unzip = True
    config.modelarts_dataset_unzip_name = 'train_data'


def main():
    '''start script for model export'''
    init_argument()
    os.environ["DEVICE_ID"] = '0'

    train.run_train()

    os.makedirs("/cache/checkpoint_path", exist_ok=True)
    mox.file.copy_parallel("/cache/train/output", "/cache/checkpoint_path")
    t = config.train_url.split("/")
    t[-1] = "output"
    config.train_url = ""
    for s in t:
        config.train_url += s + "/"
    ckpts = mox.file.list_directory('/cache/train/output', recursive=False)
    # ckpts dir
    ckpts = '/cache/train/output/' + ckpts[0]
    config.pretrained = ckpts
    ckpts = mox.file.list_directory(ckpts, recursive=False)

    # find the newest ckpt file
    last_ckpt = ''
    for ckpt in ckpts:
        if ckpt.find('.ckpt') != -1 and (last_ckpt == '' or last_ckpt < ckpt):
            last_ckpt = ckpt
    config.pretrained += '/' + last_ckpt

    export.run_export()


if __name__ == '__main__':
    main()
