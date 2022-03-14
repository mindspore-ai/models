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
"""
predict
"""
import os
from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
import numpy as np

def create_protobufVec(data, key):
    '''data'''
    data_input = MxDataInput()
    data_input.data = data.tobytes()
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for t in data.shape:
        tensorVec.tensorShape.append(t)
    tensorVec.dataStr = data_input.data
    tensorVec.tensorDataSize = len(data.tobytes())
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    return protobufVec

def infer(stream_manager, stream_name, context_id, context_pos_id,
          context_segment_id, kn_id, kn_seq_length):
    '''infer'''
    stream_manager.SendProtobuf(stream_name, b'appsrc0', create_protobufVec(context_id, b'appsrc0'))
    stream_manager.SendProtobuf(stream_name, b'appsrc1', create_protobufVec(context_segment_id, b'appsrc1'))
    stream_manager.SendProtobuf(stream_name, b'appsrc2', create_protobufVec(context_pos_id, b'appsrc2'))
    stream_manager.SendProtobuf(stream_name, b'appsrc3', create_protobufVec(kn_id, b'appsrc3'))
    stream_manager.SendProtobuf(stream_name, b'appsrc4', create_protobufVec(kn_seq_length, b'appsrc4'))

    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager.GetProtobuf(stream_name, 0, keyVec)
    if infer_result.size() == 0:
        print("inferResult is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()
    # get infer result
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    # convert the inference result to Numpy array
    score = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16).reshape(1, -1)
    score = softmax(score)

    return score

def softmax(x, axis=1):
    '''计算每行的最大值'''
    row_max = x.max(axis=axis)

    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def run_duconv(data_path, network_pipeline):
    """run duconv task"""
    context_id = np.loadtxt(data_path + '/context_id.txt', dtype=int, delimiter=' ').astype(np.int32)
    context_pos_id = np.loadtxt(data_path + '/context_pos_id.txt', dtype=int, delimiter=' ').astype(np.int32)
    context_segment_id = np.loadtxt(data_path + '/context_segment_id.txt', dtype=int, delimiter=' ').astype(np.int32)

    kn_id = np.loadtxt(data_path + '/kn_id.txt', dtype=int, delimiter=' ').astype(np.int32)
    kn_seq_length = np.loadtxt(data_path + '/kn_seq_length.txt', dtype=int, delimiter=' ').astype(np.int32)

    data_len = context_id.shape[0]

    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(network_pipeline, 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    if not os.path.exists('results'):
        os.makedirs('results')
    f = open("./results/score.txt", 'w')

    for i in range(data_len):
        output = infer(stream_manager, b"duconv", context_id[i:i+1], context_pos_id[i:i+1],\
        context_segment_id[i:i+1], kn_id[i:i+1], kn_seq_length[None, i:i+1])
        for j in output:
            f.write(str(j[1]) + '\n')
            f.flush()
    f.close()

if __name__ == '__main__':
    path = '../data/data'
    pipeline_path = '../data/config/duconv.pipeline'
    run_duconv(path, pipeline_path)
