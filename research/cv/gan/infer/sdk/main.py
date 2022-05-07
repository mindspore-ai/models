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
""" main.py """
import os
import argparse
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn, StringVector
import MxpiDataType_pb2 as MxpiDataType
import numpy as np


def parse_args(parsers):
    """
    Parse commandline arguments.
    """
    parsers.add_argument('--latent_path', type=str, default="../data/input/", help='latent path')
    parsers.add_argument('--output_path', type=str, default="../results/sdk", help='output path')
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    return parsers

def send_source_data(appsrc_id, tensors, stream_name, stream_manager_api):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
    bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    data_input = MxDataInput()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for j in tensors.shape:
        tensor_vec.tensorShape.append(j)
    array_bytes = tensors.tobytes()
    data_input.data = array_bytes
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)
    ret1 = stream_manager_api.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret1 < 0:
        print("Failed to send data to stream.")
        return False
    print("Send successfully!")
    return True


def send_appsrc_data(appsrc_id, tensors, stream_name, stream_manager1):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
    bool: send data success or not
    """
    if not send_source_data(appsrc_id, tensors, stream_name, stream_manager1):
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Om gan Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/gan.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i in range(10000):
        name = args.latent_path + "input_latent" + str(i) + ".bin"
        data = np.fromfile(name, dtype=np.float32).reshape(1, 100)
        if not send_appsrc_data(0, data, b'gan', stream_manager):
            exit()
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'gan', 0, key_vec)
        if infer_result.size() == 0:
            print("inferResult is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            exit()
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        rname = os.path.join(args.output_path, str(i) + '.bin')
        result.tofile(rname)

    # destroy streams
    stream_manager.DestroyAllStreams()
