# coding=utf-8

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" main.py """

import datetime
import json
import os
import sys
from StreamManagerApi import StreamManagerApi, MxDataInput

def info(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("[INFO][%s %d %s] %s" %(nowtime, os.getpid(), __file__, msg))

def warn(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("\033[33m[WARN][%s %d %s] %s\033[0m" %(nowtime, os.getpid(), __file__, msg))

def err(msg):
    nowtime = datetime.datetime.now().isoformat()
    print("\033[31m[ERROR][%s %d %s] %s\033[0m" %(nowtime, os.getpid(), __file__, msg))

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        err("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/resnetv2.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        err("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".png")
                or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            data_input.data = f.read()
        info("Read data from %s" % file_path)

        empty_data = []
        stream_name = b'im_resnetv2'
        in_plugin_id = 0

        start_time = datetime.datetime.now()

        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id,
                                                data_input)
        if unique_id < 0:
            err("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
        end_time = datetime.datetime.now()
        info('sdk run time: {}us'.format((end_time - start_time).microseconds))
        if infer_result.errorCode != 0:
            err("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" %
                (infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        infer_res = infer_result.data.decode()
        info("process img: {}, infer result: {}".format(file_name, infer_res))
        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiClass') is None:
            with open(res_dir_name + "/" + file_name.split('.')[0] + '.txt',
                      'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict.get('MxpiClass')
        with open(res_dir_name + "/" + file_name.split('.')[0] + '_1.txt',
                  'w') as f_write:
            res_list = [str(item.get("classId")) + " " for item in res_vec]
            f_write.writelines(res_list)
            f_write.write('\n')

    # destroy streams
    stream_manager_api.DestroyAllStreams()
