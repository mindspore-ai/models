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
"""main process for sdk infer"""
import argparse
import time
import scipy.io as sio
import numpy as np
from SdkApi import SdkApi

STREAM_NAME = b'dem'
TENSOR_DTYPE_FLOAT16 = 1
TENSOR_DTYPE_FLOAT32 = 0


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="dem process")
    parser.add_argument("--pipeline_path", type=str, default="../pipeline/dem.pipeline", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="/dataset/DEM_data", help="path where the dataset is saved")
    parser.add_argument("--dataset", type=str, default="AwA", choices=['AwA', 'CUB'],
                        help="dataset which is chosen to use")
    args_opt = parser.parse_args()
    return args_opt


def inference():
    """infer process function"""
    args = parse_args()

    # init stream manager
    sdk_api = SdkApi(args.pipeline_path)
    if not sdk_api.init():
        exit(-1)

    start_time = time.time()
    if args.dataset == 'AwA':
        input_tensor = dataset_AwA(args.data_dir)
    elif args.dataset == 'CUB':
        input_tensor = dataset_cub(args.data_dir)
    print("================> Input shape:", input_tensor.shape)
    res_list = np.empty((input_tensor.shape[0], 1, 1024), dtype=np.float32)
    for i in range(input_tensor.shape[0]):
        input_data = input_tensor[i]
        sdk_api.send_tensor_input(STREAM_NAME, 0, b'appsrc0', input_data, input_tensor.shape, TENSOR_DTYPE_FLOAT32)
        result = sdk_api.get_result(STREAM_NAME)
        pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        res_list[i] = pred
    end_time = time.time() - start_time
    print(f"The inference time is {end_time}")
    res = np.squeeze(res_list, axis=1)
    # save result
    np.savetxt('res', res, fmt="%f")


def dataset_cub(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/CUB_data/test_proto.mat')
    test_att_0 = np.array(f['test_proto'])
    test_att_0 = test_att_0.astype("float32")

    return test_att_0


def dataset_AwA(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/AwA_data/attribute/pca_te_con_10x85.mat')
    test_att_0 = np.array(f['pca_te_con_10x85'])
    test_att_0 = test_att_0.astype("float32")

    return test_att_0


if __name__ == '__main__':
    inference()
