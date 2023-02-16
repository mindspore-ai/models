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
""" preprocess of LR-EGAN model """

import argparse
from dataloader import LoadDocumentData, LoadImageData, LoadTabularData, LoadMatData


class parameter():
    ''' parammeter '''
    def __init__(self):
        parser = argparse.ArgumentParser()
        # data parameter
        parser.add_argument('--data_format', type=str, default='csv',
                            help='Dataset format')
        parser.add_argument('--data_name', type=str, default='market',
                            help='Dataset name')
        parser.add_argument('--data_path', type=str, default='./data/',
                            help='Data path')
        parser.add_argument('--inject_noise', type=int, default=0,
                            help='Whether to inject noise to train data')
        parser.add_argument('--cont_rate', type=float, default=0,
                            help='Inject noise to contamination rate')
        parser.add_argument('--anomal_rate', type=str, default='default',
                            help='Adjust anomaly rate')
        parser.add_argument('--seed', type=int, default=42,
                            help="Random seed.")
        parser.add_argument('--verbose', action='store_false', default=True,
                            help='Whether to print training details')

        if __name__ == '__main__':
            args = parser.parse_args()
        else:
            args = parser.parse_args([])

        self.__dict__.update(args.__dict__)


def preprocess(args=None):
    ''' preprocess '''
    if args is None:
        args = parameter()
    # tabular

    if args.data_name in ['attack', 'bcsc', 'creditcard', 'diabetic', 'donor', 'intrusion', 'market', 'thyroid']:
        x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(args)
    # document
    elif args.data_name in ['20news', 'reuters']:
        dataloader = LoadDocumentData(args)
        if args.anomal_rate == 'default':
            args.anomal_rate = 0.05

        for normal_idx in range(dataloader.class_num):
            x_train, x_test, y_train, y_test = dataloader.preprocess(
                normal_idx)
            x_val = x_train.copy()
            y_val = y_train.copy()
    # image
    elif args.data_name in ['mnist']:
        x_train, x_test, y_train, y_test = LoadImageData(args)
        x_val = x_train.copy()
        y_val = y_train.copy()

    elif args.data_format == "mat":
        x_train, y_train, x_val, y_val, x_test, y_test = LoadMatData(args)

    return x_train, y_train, x_val, y_val, x_test, y_test
