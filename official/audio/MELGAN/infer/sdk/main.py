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
import argparse

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput


def inference(input_tensor):
    tensor_bytes = input_tensor.tobytes()
    in_plugin_id = 0
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    dataInput = MxDataInput()
    dataInput.data = tensor_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for t in input_tensor.shape:
        tensorVec.tensorShape.append(t)
    tensorVec.dataStr = dataInput.data
    tensorVec.tensorDataSize = len(tensor_bytes)
    # add feature data end
    key = "appsrc{}".format(in_plugin_id).encode('utf-8')
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)
    unique_id = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobufVec)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()
    # Obtain the inference result by specifying streamName and uniqueId.
    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
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
    out = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).ravel()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, default='../data/input/',
                        help="input data path")
    parser.add_argument('--pipeline_path', type=str, default='./output',
                        help='pipeline path')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='output data path')
    parser.add_argument('--eval_length', type=int, default=240,
                        help='eval length')
    parser.add_argument('--hop_size', type=int, default=256,
                        help='hop size')
    parser.add_argument('--sample', type=int, default=22050,
                        help='sample')
    opts = parser.parse_args()
    eval_path = opts.eval_path
    output_path = opts.output_path
    pipeline_path = opts.pipeline_path
    eval_length = opts.eval_length
    hop_size = opts.hop_size
    sample = opts.sample
    repeat_frame = eval_length // 8

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Construct the input of the stream
    infer_total_time = 0
    files = os.listdir(eval_path)
    for file_name in files:
        if "_test.txt" in file_name:
            data_path = os.path.join(eval_path, file_name)
            all_test_data = np.loadtxt(data_path, dtype=np.float32)
            stream_name = b'im_melgan'
            all_test_data = all_test_data.reshape((-1, 1, 80, 240))
            num = all_test_data.shape[0]

            # first frame
            wav_data = np.array([])
            tensor = all_test_data[0].reshape((1, 80, 240))
            for idx in range(0, num):
                tensor = all_test_data[idx].reshape((1, 80, 240))
                output = inference(tensor)
                wav_data = np.concatenate((wav_data, output))

            # save as txt file
            out_path = os.path.join(output_path, 'restruction_' + file_name)
            np.savetxt(out_path, wav_data.reshape(-1), fmt='%.18e')
            print("File " + file_name + " inference successfully!")

    # destroy streams
    stream_manager_api.DestroyAllStreams()
