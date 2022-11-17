# coding=utf-8

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

import os
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    Args:
        appsrc_id: an RGB image:the appsrc component number for SendProtobuf
        tensor: the tensor type of the input file
        stream_name: stream Name
        stream_manager:the StreamManagerApi
    Returns:
        bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
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


def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("../data/config/transformer.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
    stream_name = b'transformer'
    predictions = []
    path = '../data/data/00_source_eos_ids'
    path1 = '../data/data/01_source_eos_mask'
    files = os.listdir(path)
    for i in range(len(files)):
        full_file_path = os.path.join(path, "transformer_bs_1_" + str(i) + ".bin")
        full_file_path1 = os.path.join(path1, "transformer_bs_1_" + str(i) + ".bin")
        source_ids = np.fromfile(full_file_path, dtype=np.int32)
        source_mask = np.fromfile(full_file_path1, dtype=np.int32)
        source_ids = np.expand_dims(source_ids, 0)
        source_mask = np.expand_dims(source_mask, 0)
        if not send_source_data(0, source_ids, stream_name, stream_manager_api):
            return
        if not send_source_data(1, source_mask, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)
        predictions.append(res.reshape(1, 1, 81))
    # decode and write to file
    f = open('./results', 'w')
    for batch_out in predictions:
        token_ids = [str(x) for x in batch_out[0][0].tolist()]
        f.write(" ".join(token_ids) + "\n")
    f.close()
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
