#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2022 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import StreamManagerApi.py
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector

def send_source_data(fig, appsrc_id, filename, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    with open(filename, 'r') as file:
        tensor = file.readlines()
    tensor0 = []
    tensor1 = []
    tensor0 = tensor[0]
    tensor0 = tensor0.split(',')
    tensor1.append(tensor0[fig])
    tensor = np.asarray(tensor1, dtype=np.int32)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True

def send_source_data1(fig, appsrc_id, filename, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    with open(filename, 'r') as f:
        tensor = f.readlines()
    tensor0 = []
    tensor1 = []
    tensor0 = tensor[0]
    tensor0 = tensor0.split(',')
    for k in range(5):
        tensor1.append(tensor0[k])
    tensor = np.asarray(tensor1, dtype=np.int32).reshape(1, 5)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for tenvec in tensor.shape:
        tensor_vec.tensorShape.append(tenvec)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret2 = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret2 < 0:
        print("Failed to send data to stream.")
        return False
    return True

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret0 = streamManagerApi.InitManager()
    if ret0 != 0:
        print("Failed to init Stream manager, ret0=%s" % str(ret0))
        exit()

    # create streams by pipeline config file
    with open("../data/config/skipgram.pipeline", 'rb') as pip:
        pipelineStr = pip.read()
    ret1 = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret1 != 0:
        print("Failed to create Stream, ret1=%s" % str(ret1))
        exit()

    with open("./test1.txt", 'r') as s:
        center = s.readlines()
    word = []
    data = []
    word = center[0]
    word = word.split(',')
    print("word = ", word, type(word), len(word))
    num = 0
    for leni in range(len(word)):
        num = leni
        # Construct the input of the stream
        streamName = b'transfer'
        send_source_data(num, 0, "./test1.txt", streamName, streamManagerApi)
        send_source_data(num, 1, "./test2.txt", streamName, streamManagerApi)
        send_source_data1(num, 2, "./test3.txt", streamName, streamManagerApi)
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        inferResult = streamManagerApi.GetProtobuf(streamName, 0, key_vec)
        if inferResult.size() == 0:
            print("inferResult is null")
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(inferResult[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        data.append(res)
    # print the infer result
    print("data = ", data)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
