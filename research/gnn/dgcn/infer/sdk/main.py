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
""" Model Main """
import argparse
import time
import glob
import numpy as np
from api.infer import SdkApi
from config import config as cfg


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--pipeline_path", type=str, default="./config/dgcn.pipeline", help="SDK infer pipeline")
    parser.add_argument("--dataset", type=str, default="cora", help="dataset name")
    parser.add_argument("--data_dir", type=str, default="../data/input/cora_bin/",
                        help="Dataset contain batch_spare batch_label batch_dense")
    args_opt = parser.parse_args()
    return args_opt


def inference(pipeline_path, stream_name, dataset):
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    files1 = glob.glob(pathname=args.data_dir + '/00_data/' + '*.bin')
    files2 = glob.glob(pathname=args.data_dir + '/01_data/' + '*.bin')
    files3 = glob.glob(pathname=args.data_dir + '/02_data/' + '*.bin')
    files_length1 = len(files1)
    files_length2 = len(files2)
    files_length3 = len(files3)
    if dataset == "cora":
        number = 2708
        number1 = 1433
        number2 = 7
        filename = 'pred1.txt'
    if dataset == "citeseer":
        number = 3327
        number1 = 3703
        number2 = 6
        filename = 'pred2.txt'
    if dataset == "pubmed":
        number = 19717
        number1 = 500
        number2 = 3
        filename = 'pred3.txt'
    for i in range(files_length1):
        start_time = time.time()
        input_spare_file = args.data_dir + "/00_data/" + "diffusions.bin"
        input_spare = np.fromfile(input_spare_file, dtype=np.float16).reshape(number, number)
        sdk_api.send_tensor_input(stream_name, 0, "appsrc0", input_spare, input_spare.shape, cfg.TENSOR_DTYPE_FLOAT16,
                                  number)
    for i in range(files_length2):
        input_dense_file = args.data_dir + "/01_data/" + "ppmi.bin"
        input_dense = np.fromfile(input_dense_file, dtype=np.float16).reshape(number, number)
        sdk_api.send_tensor_input(stream_name, 1, "appsrc1", input_dense, input_dense.shape, cfg.TENSOR_DTYPE_FLOAT16,
                                  number)
    for i in range(files_length3):
        # set label data
        input_label_file = args.data_dir + "/02_data/" + "feature.bin"
        input_label = np.fromfile(input_label_file, dtype=np.float16).reshape(number, number1)
        sdk_api.send_tensor_input(stream_name, 2, "appsrc2", input_label, input_label.shape, cfg.TENSOR_DTYPE_FLOAT16,
                                  number)
    result = sdk_api.get_result(stream_name)
    end_time = time.time() - start_time
    print(f"The criteo({i}) inference time is {end_time}")
    pred = np.array(
        [np.frombuffer(result.tensorPackageVec[i].tensorVec[0].dataStr, dtype=np.float32) for i in range(number)])
    with open(filename, 'a+') as f:
        for i in range(number):
            for j in range(number2):
                f.write(str(pred[i][j]) + " ")
            f.write("\n")

if __name__ == "__main__":
    args = parse_args()
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    inference(args.pipeline_path, args.stream_name, args.dataset)
