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
from StreamManagerApi import MxDataInput, InProtobufVector,\
    MxProtobufIn, StringVector, MxBufferInput, MxMetadataInput, MetadataInputVector, StreamManagerApi
import MxpiDataType_pb2 as MxpiDataType
import numpy as np


def getData(path):
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(128, -1)
    return data


def send_source_data(filename, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    stream_name = b'tall'
    tensor = getData(filename)
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
    key = "appsrc0".encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)
    ret2 = stream_manager.SendProtobuf(stream_name, 0, protobuf_vec)
    if ret2 < 0:
        print("Failed to send data to stream.")
        return False
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager.GetProtobuf(stream_name, 0, key_vec)
    if infer_result.size() == 0:
        print("inferResult is null")
        return False
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
        return False
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    return res


def infer(img_path, streamManager):
    # Construct the input of the stream
    data_input = MxDataInput()
    data_128 = getData(img_path)
    data_input.data = data_128.tobytes()
    # Inputs data to a specified stream based on streamName.
    batch_size = 128
    stream_name = b'tall'
    elment_name = b'appsrc0'
    key = b'mxpi_tensorinfer0'
    frame_info = MxpiDataType.MxpiFrameInfo()
    frame_info.frameId = 0
    frame_info.channelId = 0

    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionData.deviceId = 0
    vision_vec.visionData.memType = 0
    vision_vec.visionData.dataStr = data_input.data

    buffer_input = MxBufferInput()
    buffer_input.mxpiFrameInfo = frame_info.SerializeToString()
    buffer_input.mxpiVisionInfo = vision_vec.SerializeToString()
    buffer_input.data = data_input.data

    metedata_input = MxMetadataInput()
    metedata_input.dataSource = elment_name
    metedata_input.dataType = b"MxTools.MxpiVisionList"
    metedata_input.serializedMetadata = vision_list.SerializeToString()

    metedata_vec = MetadataInputVector()
    metedata_vec.push_back(metedata_input)

    error_code = streamManager.SendData(stream_name, elment_name, metedata_vec, buffer_input)

    if error_code < 0:
        print("Failed to send data to stream.")
        exit()
    data_source_vector = StringVector()
    data_source_vector.push_back(key)
    infer_result = streamManager.GetResult(stream_name, b'appsink0', data_source_vector)
    infer_result = infer_result.bufferOutput.data
    infer_result1 = infer_result.metadataVec[0]
    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result1.serializedMetadata)
    print(tensorList.tensorPackageVec[0].tensorVec[0].tensorShape)
    nparr = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    nparr = nparr.reshape((batch_size, batch_size, 3))
    return nparr


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret1 = streamManagerApi.InitManager()
    if ret1 != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret1))
        exit()

    # create streams by pipeline config file
    with open("./config/tall.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    with open('infer.txt', 'r', encoding='utf-8') as f:
        datapaths = f.readlines()
    if not os.path.exists('./result'):
        os.makedirs('./result')
    for datapath in datapaths:
        output = send_source_data(datapath[:-1], streamManagerApi)
        dataname = datapath[:-1].split('/')[-1].replace('.data', '.bin')
        output.tofile(f'./result/{dataname}')
    # destroy streams
    streamManagerApi.DestroyAllStreams()
