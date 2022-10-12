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
"""Configuration for FastFlow"""
import argparse

def get_arguments():
    """Configurations"""
    parser = argparse.ArgumentParser(description='MindSpore FastFlow')

    # dataset configurations:
    parser.add_argument('--dataset_path', type=str, default="/data/mvtec",
                        help='the path of mvtec dataset')
    parser.add_argument('--category', type=str, default="bottle",
                        help='the category of mvtec dataset')
    parser.add_argument('--mean', type=list, default=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                        help='mean of mvtec dataset, computed from random subset of ImageNet training images')
    parser.add_argument('--std', type=list, default=[1/0.229, 1/0.224, 1/0.255],
                        help='std of mvtec dataset, computed from random subset of ImageNet training images')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help="batch size when training")
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help="batch size when testing")

    # model configurations:
    parser.add_argument('--pre_ckpt_path', type=str, default="pretrained/wide_resnet50_racm-8234f177.ckpt",
                        help='pretrained feature extractor for fastflow')
    parser.add_argument('--im_resize', type=int, default=256,
                        help="the size of resized images for model")
    parser.add_argument('--flow_step', type=int, default=8,
                        help="steps of fastflow")
    parser.add_argument('--hidden_ratio', type=float, default=1.0,
                        help="hidden ratio in subset of fastflow")
    parser.add_argument('--conv3x3_only', type=bool, default=False,
                        help="only conv3x3 in subset of fastflow ")

    # train/test configurations:
    parser.add_argument('--num_epochs', type=int, default=500,
                        help="the number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight rate")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="interval steps for log when training")
    parser.add_argument('--eval_interval', type=int, default=10,
                        help="interval epochs for evaluation when training")
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt',
                        help="the directory of ckpt file")
    parser.add_argument('--ckpt_path', type=str, default='',
                        help="the path of ckpt file")
    parser.add_argument('--save_imgs', type=bool, default=False,
                        help="visilize imgs when testing or not")
    return parser
