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

import argparse

def get_train_argparser():
    """command parser for train"""
    parser = argparse.ArgumentParser()

    # general argument
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('--device', type=str,
                        help='[CPU/Ascend/GPU]')

    parser.add_argument('--num_features', type=int, default=64,
                        help='num discrete features [%(default)d]')
    parser.add_argument('--num_neighbors', type=int, default=10,
                        help='num neighbors [%(default)d]')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size [%(default)d]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--epochs', type=int, default=100,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--num_retrieve', type=int, default=100,
                        help='num neighbors to retrieve [%(default)d]')
    parser.add_argument('--num_bad_epochs', type=int, default=6,
                        help='num indulged bad epochs [%(default)d]')
    parser.add_argument('--distance_metric', default='hamming',
                        choices=['hamming', 'cosine'])

    # model specific argument
    parser.add_argument('--num_trees', type=int, default=10,
                        help='num of trees [%(default)d]')
    parser.add_argument("--temperature", type=float, default=0.1,
                        help='temperature for binarization [%(default)g]')
    parser.add_argument("--alpha", type=float, default=0.1,
                        help='temperature for sampling neighbors [%(default)g]')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='beta term (as in beta-VAE) [%(default)g]')
    return parser

def get_eval_argparser():
    """command parser for eval"""
    parser = argparse.ArgumentParser()
    parser.add_argument('hparams_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--device', type=str, help='[CPU/Ascend/GPU]')
    return parser

def get_export_argparser():
    """command parser for export"""
    parser = argparse.ArgumentParser()
    parser.add_argument('hparams_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--device', type=str, help='[CPU/Ascend/GPU]')
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--file_format', type=str, choices=['AIR', 'MINDIR'], default='MINDIR', help='file format')
    return parser
