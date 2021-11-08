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
import argparse
import os
import glob

import moxing as mox
import export
import train
from model_utils.config import config


def parse_args():
    """get the cmd input args"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_url', type=str, default='', help='the path model saved')
    parser.add_argument('-d', '--data_url', type=str, default='', help='the training data')

    parser.add_argument("--device_id", type=str, default="0", help="device id")
    parser.add_argument("--max_epoch", type=int, default=40, help="max epoch")
    return parser.parse_args()


def main():
    """start script for model training and exporting"""
    args = parse_args()
    print("Training setting:", args)
    os.environ["DEVICE_ID"] = args.device_id
    config.enable_modelarts = True
    dataset_list = glob.glob(os.path.join(args.data_url, '*.zip'))
    config.modelarts_dataset_unzip_name = dataset_list[0].split("/")[-1].split(".")[0]
    label_file_list = glob.glob(os.path.join(args.data_url, '*.txt'))
    config.train_label_file = label_file_list[0]
    config.max_epoch = args.max_epoch
    train.run_train()
    os.makedirs(config.load_path, exist_ok=True)
    mox.file.copy_parallel(os.path.join(config.output_path, 'output'), config.load_path)
    config.pretrained = glob.glob(os.path.join(config.load_path, '*/*.ckpt'))[0]
    config.train_url = os.path.join(config.train_url, "model")
    os.makedirs(config.train_url, exist_ok=True)
    export.run_export()


if __name__ == '__main__':
    main()
