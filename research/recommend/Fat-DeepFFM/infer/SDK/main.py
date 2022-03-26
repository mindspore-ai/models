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
import glob
import time

import numpy as np
from api.infer import SdkApi
from config import config as cfg


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--pipeline_path", type=str, default="./config/fat_deepffm.pipeline", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="../data/input/Criteo_bin/",
                        help="Dataset contain batch_spare batch_label batch_dense")
    args_opt = parser.parse_args()
    return args_opt


def criteo_inference(stream_name):
    """model inference"""
    args = parse_args()
    # init stream manager
    sdk_api = SdkApi(args.pipeline_path)
    if not sdk_api.init():
        exit(-1)

    files = glob.glob(pathname=args.data_dir + 'batch_spare/*.bin')
    files_length = len(files)

    for i in range(files_length):
        start_time = time.time()
        # set sparse data
        input_spare_file = args.data_dir + "batch_spare/criteo_bs1000_" + str(i) + ".bin"
        input_spare = np.fromfile(input_spare_file, dtype=np.int32).reshape(1000, 26)
        print(input_spare.shape)
        sdk_api.send_tensor_input(stream_name, 0, "appsrc0", input_spare, input_spare.shape, cfg.TENSOR_DTYPE_INT32)

        # set dense data
        input_dense_file = args.data_dir + "batch_dense/criteo_bs1000_" + str(i) + ".bin"
        input_dense = np.fromfile(input_dense_file, dtype=np.float32).reshape(1000, 13)
        print(input_dense.shape)
        sdk_api.send_tensor_input(stream_name, 1, "appsrc1", input_dense, input_dense.shape, cfg.TENSOR_DTYPE_FLOAT32)

        # set label data
        input_label_file = args.data_dir + "batch_labels/criteo_bs1000_" + str(i) + ".bin"
        input_label = np.fromfile(input_label_file, dtype=np.float32).reshape(1000, 1)
        print(input_label.shape)
        sdk_api.send_tensor_input(stream_name, 2, "appsrc2", input_label, input_label.shape, cfg.TENSOR_DTYPE_FLOAT32)

        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time
        print(f"The criteo({i}) inference time is {end_time}")
        pred = np.array(
            [np.frombuffer(result.tensorPackageVec[k].tensorVec[0].dataStr, dtype=np.float32) for k in range(1000)])
        with open('pred.txt', 'a+') as f:
            for m in range(1000):
                f.write(str(pred[m][0]) + " ")
            f.write("\n")

        with open('label.txt', 'a+') as p:
            for j in range(1000):
                p.write(str(input_label[j][0]) + " ")
            p.write("\n")


if __name__ == "__main__":
    criteo_inference(cfg.STREAM_NAME.encode("utf-8"))
