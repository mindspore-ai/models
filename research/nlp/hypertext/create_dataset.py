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
preprocess corpus and obtain mindrecord file.
"""

import argparse
import os

from src.data_preprocessing import changeIflytek, changeTnews

parser = argparse.ArgumentParser(description='preprocess corpus and obtain mindrecord.')
parser.add_argument('--data_dir', type=str, default='/data/tnews/', help='the directory of data.')
parser.add_argument('--out_data_dir', type=str, default='/data/tnews_public/',
                    help='the directory of file processing output ')
parser.add_argument('--datasetType', default='tnews', type=str, help='iflytek/tnews')
args = parser.parse_args()


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


create_dir_not_exist(args.out_data_dir)
args.data_dir = os.path.abspath(os.path.realpath(args.data_dir))
args.out_data_dir = os.path.abspath(os.path.realpath(args.out_data_dir))

if args.datasetType == 'iflytek':
    changeIflytek(args.data_dir, args.out_data_dir)
if args.datasetType == 'tnews':
    changeTnews(args.data_dir, args.out_data_dir)
