# coding=utf-8
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

import datetime
import os
import sys
import numpy as np
import cv2

from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput
from StreamManagerApi import StringVector
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import InProtobufVector
import MxpiDataType_pb2 as MxpiDataType

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./mcnn.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]
    gt_name = sys.argv[2]

    file_list = os.listdir(dir_name)
    file_list.sort()
    mae = 0
    mse = 0
    start_time = datetime.datetime.now()
    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        gt_path = os.path.join(gt_name, file_name[:-3] + 'csv')
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg") \
                or file_name.lower().endswith(".png")):
            continue

        empty_data = []
        stream_name = b'mcnn_opencv'
        in_plugin_id = 0
        input_key = 'appsrc0'

        img = cv2.imread(file_path, 0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        hang_left = 512 - int(ht / 2)
        hang_right = 1024 - hang_left - ht
        lie_left = 512 - int(wd / 2)
        lie_right = 1024 - lie_left - wd
        img = np.pad(img, ((hang_left, hang_right), (lie_left, lie_right)), 'constant')

        img = img.reshape((1, 1, 1024, 1024))
        tensor_list = MxpiDataType.MxpiTensorPackageList()
        tensor_pkg = tensor_list.tensorPackageVec.add()

        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(img.shape)
        tensor_vec.tensorDataType = 0
        tensor_vec.dataStr = img.tobytes()
        tensor_vec.tensorDataSize = len(img)
        buf_type = b"MxTools.MxpiTensorPackageList"

        protobuf = MxProtobufIn()
        protobuf.key = input_key.encode("utf-8")
        protobuf.type = buf_type
        protobuf.protobuf = tensor_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)
        err_code = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
        if err_code != 0:
            print(
                "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
                "buf_type(%s), err_code(%s).", stream_name, in_plugin_id,
                input_key, buf_type, err_code)

        keys = [b"mxpi_tensorinfer0",]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        TensorList = MxpiDataType.MxpiTensorPackageList()
        TensorList.ParseFromString(infer_result[0].messageBuf)
        data = np.frombuffer(TensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

        den = np.loadtxt(open(gt_path, "rb"), delimiter=",", skiprows=0)
        den = den.astype(np.float32, copy=False)
        gt_count = np.sum(den)
        et_count = np.sum(data)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
        print(file_path, "True value:", np.sum(den), "predictive value:", np.sum(data))

    mae = mae / 182
    mse = np.sqrt(mse / 182)
    end_time = datetime.datetime.now()
    print("*********************************************")
    print("Final accuracy of the project:")
    print('MAE:', mae, '  MSE:', mse)
    print("*********************************************")
    print("Overall project performance:")
    print(182 / (end_time - start_time).seconds, "images/seconds")

    # destroy streams
    stream_manager_api.DestroyAllStreams()
