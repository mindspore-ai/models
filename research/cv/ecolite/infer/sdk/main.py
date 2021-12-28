# -*- coding:utf-8 -*-
# Copyright(C) 2021. Huawei Technologies Co.,Ltd
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
Use this file for sdk running
"""
import os
import time
import argparse
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector
import MxpiDataType_pb2 as MxpiDataType
import numpy as np


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='get ecolite train dataset')
    parser.add_argument('type', type=str, default="1", choices=['1', '2'],
                        help='1 is used for single record infer, 2 is used for full data infer')
    parser.add_argument('root_data_dir', type=str)
    parser.add_argument('batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    return parser.parse_args()


args = get_args()


def getData(path):
    """ get data"""
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(args.batch_size, 12, 224, 224)
    return data


def send_and_infer(filename, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    stream_name = b'ecolite'
    tensor = getData(filename)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i_ in tensor.shape:
        tensor_vec.tensorShape.append(i_)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc0".encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret_ = stream_manager.SendProtobuf(stream_name, 0, protobuf_vec)
    if ret_ < 0:
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

    result_ = MxpiDataType.MxpiTensorPackageList()
    result_.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result_.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')

    res = res.reshape((16, 101))
    return res


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./config/ecolite.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    time_begin = time.time()
    if args.type == '2':
        if not os.path.exists('./result'):
            os.makedirs('./result')
        root_dir = args.root_data_dir
        datapath = os.listdir(root_dir)
        for i, path_ in enumerate(datapath):
            datapath = os.path.join(root_dir, path_)
            result = send_and_infer(datapath, streamManagerApi)
            videoid = datapath.split('/')[-1].split('_')[-2]
            dataname = 'eval_predict_' + str(videoid) + '_.bin'
            result.tofile(f'./result/{dataname}')
    if args.type == '1':
        if not os.path.exists('./result_single'):
            os.makedirs('./result_single')
        dirname = args.root_data_dir
        datapath = os.listdir(dirname)
        datapath = os.path.join(dirname, datapath[-1])
        result = send_and_infer(datapath, streamManagerApi)
        videoid = datapath.split('/')[-1].split('_')[-2]
        dataname = 'eval_predict_' + str(videoid) + '_.bin'
        result.tofile(f'./result_single/{dataname}')
    time_end = time.time()
    print("infer time cost:", time_end - time_begin)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
