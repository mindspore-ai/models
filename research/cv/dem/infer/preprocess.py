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
"""preprocessing dataset for mx-base infer"""

import argparse
import os
import scipy.io as sio
import numpy as np


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="dem process")
    parser.add_argument("--data_dir", type=str, default="/home/dataset/DEM_data/",
                        help="path where the dataset is saved")
    args_opt = parser.parse_args()
    return args_opt


def dataset_cub(data_path):
    f = sio.loadmat(data_path+'/CUB_data/test_proto.mat')
    test_att_0 = np.array(f['test_proto'])
    test_att_0 = test_att_0.astype("float32")
    os.mkdir(data_path + "/CUB_data/test_att")
    for i in range(test_att_0.shape[0]):
        np.savetxt(data_path + '/CUB_data/test_att/test_att_%d' % i, test_att_0[i], fmt="%f")


def dataset_AwA(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/AwA_data/attribute/pca_te_con_10x85.mat')
    test_att_0 = np.array(f['pca_te_con_10x85'])
    test_att_0 = test_att_0.astype("float32")
    os.mkdir(data_path + "/AwA_data/test_att")
    for i in range(test_att_0.shape[0]):
        np.savetxt(data_path + '/AwA_data/test_att/test_att_%d' % i, test_att_0[i], fmt="%f")
    return test_att_0


if __name__ == '__main__':
    args = parse_args()
    dataset_cub(args.data_dir)
    dataset_AwA(args.data_dir)
