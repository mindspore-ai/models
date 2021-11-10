'''
The scripts to execute sdk infer
'''
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


import argparse
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="protonet process")
    parser.add_argument("--pipeline", type=str, default="", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--infer_result_path", type=str, default="")
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

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

    args = parse_args()
    BATCH_SIZE = 100

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'protonet'
    infer_total_time = 0
    file_list = os.listdir(args.data_dir)
    infer_result_folder = args.infer_result_path
    for file_name in file_list:
        num = file_name.split('_')[1]
        file_path = os.path.join(args.data_dir, file_name)
        tensor = np.fromfile(file_path, dtype=np.float32)
        tensor = np.resize(tensor, (100, 1, 28, 28))
        array_list = []
        for tensor0 in tensor:
            tensor0 = tensor0.reshape((1, 1, 28, 28))
            if not send_source_data(0, tensor0, stream_name, stream_manager_api):
                return

            # Obtain the inference result by specifying streamName and uniqueId.
            key_vec = StringVector()
            key_vec.push_back(b'tensorinfer0')
            start_time = time.time()
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            infer_total_time += time.time() - start_time
            if infer_result.size() == 0:
                print("inferResult is null")
                return
            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
                return
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            res = res.reshape((1, 64))
            array_list.append(res)

        tensor = np.vstack((array_list[0], array_list[1]))
        for i in range(BATCH_SIZE - 2):
            tensor = np.vstack((tensor, array_list[i+2]))
        tensor.tofile(infer_result_folder + "/" + "data_" + num)
    print("=======================================")
    print("The total time of inference is {} s".format(infer_total_time))
    print("=======================================")

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
