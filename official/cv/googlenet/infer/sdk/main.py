#!/usr/bin/env python

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import sys

from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput


def run():
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/googlenet.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()
    if len(sys.argv) == 3:
        dir_name = sys.argv[1]
        res_dir_name = sys.argv[2]
    else:
        print("Please enter Dataset path| Inference result path "
              "such as ../cifar10 ./result")
        exit(1)
    if not os.path.exists(dir_name):
        print("Dataset path does not exist.")
        exit(1)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
        print("Inference result path does not exist.")
    file_list = os.listdir(dir_name)

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue
        with open(file_path, 'rb') as f:
            data_input.data = f.read()
        stream_name = b'im_googlenet'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        infer_res = infer_result.data.decode()
        print("process img: {}, infer result: {}".format(file_name, infer_res))
        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiClass') is None:
            with open(res_dir_name + "/" + file_name[:-5] + '.txt', 'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict.get('MxpiClass')
        res_name = os.path.join(res_dir_name, '{}_1.txt'.format(file_name[:-4]))
        with open(res_name, 'w') as f_write:
            res_list = [str(item.get("classId"))+ " " for item in res_vec]
            f_write.writelines(res_list)
            f_write.write('\n')

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    run()
