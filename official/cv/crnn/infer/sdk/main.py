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

import argparse
import os
import json
from StreamManagerApi import StreamManagerApi, MxDataInput


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="crnn process")
    parser.add_argument("--dir_name", type=str,
                        default=None, help="infer input dir")
    parser.add_argument('--res_dir_name', default='./output', type=str,
                        help='the folder to save the result')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':

    args = parse_args()
    dir_name = args.dir_name
    res_dir_name = args.res_dir_name

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    pipeline_path = "../config/crnn.pipeline"
    stream_name = b'crnn'
    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Input data to a specified stream based on streamName
    in_plugin_id = 0
    # Construct the input of the stream
    data_input = MxDataInput()
    file_list = os.listdir(dir_name)

    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        print(file_name)
        if file_name.endswith(".JPG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
            file_path = os.path.join(dir_name, file_name)
            with open(file_path, 'rb') as f:
                data_input.data = f.read()
                print("input")
                unique_id = stream_manager_api.SendDataWithUniqueId(
                    stream_name, in_plugin_id, data_input)
                if unique_id < 0:
                    print("Failed to send data to stream.")
                    exit()
                infer_result = stream_manager_api.GetResultWithUniqueId(
                    stream_name, unique_id, 3000)
                if infer_result.errorCode != 0:
                    print("GetResultWithUniqueId error. errorCode=%d, "
                          "errorMsg=%s" % (infer_result.errorCode,
                                           infer_result.data.decode()))
                    exit()
                print("get out")
                # print the infer result
                print(infer_result.data.decode())
                res_dict = json.loads(infer_result.data.decode())
                save_path = res_dir_name + "/" + file_name[:-4] + ".json"
                with open(save_path, "w") as f:
                    json.dump(res_dict, f, sort_keys=True, indent=4,
                              separators=(',', ':'))
    # destroy streams
    stream_manager_api.DestroyAllStreams()
