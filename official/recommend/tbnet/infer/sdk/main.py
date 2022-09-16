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
import argparse
import os
from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
import numpy as np


def parse_args(parsers):
    """
    Parse commandline arguments.
    """
    parsers.add_argument('--data_path', type=str,
                         default="../../preprocess_Result",
                         help='text path')
    return parsers

def create_protobuf(path, id1, shape):
    # Construct the input of the stream
    data_input = MxDataInput()
    with open(path, 'rb') as f:
        data = f.read()
    data_input.data = data
    tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
    tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
    tensorVec1 = tensorPackage1.tensorVec.add()
    tensorVec1.deviceId = 0
    tensorVec1.memType = 0
    for t in shape:
        tensorVec1.tensorShape.append(t)
    tensorVec1.dataStr = data_input.data
    tensorVec1.tensorDataSize = len(data)

    protobuf1 = MxProtobufIn()
    protobuf1.key = b'appsrc%d' % id1
    protobuf1.type = b'MxTools.MxpiTensorPackageList'
    protobuf1.protobuf = tensorPackageList1.SerializeToString()

    return protobuf1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Om tbnet Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/tbnet.pipeline", 'rb') as fl:
        pipeline = fl.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream

    res_dir_name = 'result'
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    results = []
    input_names = ['00_item', '01_rl1', '02_ety', '03_rl2', '04_his', '05_rate']
    shape_list = [[1], [1, 39], [1, 39], [1, 39], [1, 39], [1]]

    for idx in range(18415):
        print('infer %d' % idx)
        for index, name in enumerate(input_names):
            protobufVec = InProtobufVector()
            path_tmp = os.path.join(args.data_path, name,
                                    'tbnet_' + name.split('_')[1] + '_bs1_' + str(idx) + '.bin')
            protobufVec.push_back(create_protobuf(path_tmp, index, shape_list[index]))
            unique_id = stream_manager.SendProtobuf(b'tbnet', b'appsrc%d' % index, protobufVec)

        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'tbnet', 0, keyVec)
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
        for i in range(4):
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[i].dataStr, dtype=np.float32)
            np.savetxt("./result/tbnet_item_bs1_%d_%d.txt" % (idx, i), res, fmt='%.06f')

    # destroy streams
    stream_manager.DestroyAllStreams()
