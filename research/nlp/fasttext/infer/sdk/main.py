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
sample script of CLUE infer using SDK run in docker
"""

import argparse
import os

import datetime
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="Mass process")
    parser.add_argument("--pipeline", type=str, default="../data/config/fasttext.pipline", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="../data/input")
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(tensor, tensor_bytes, name, manager_api, in_plugin_id):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    dataInput = MxDataInput()
    dataInput.data = tensor_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for t in tensor.shape:
        tensorVec.tensorShape.append(t)
    tensorVec.dataStr = dataInput.data
    tensorVec.tensorDataSize = len(tensor_bytes)
    key = "appsrc{}".format(in_plugin_id).encode('utf-8')
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)
    unique_id = manager_api.SendProtobuf(name, in_plugin_id, protobufVec)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

def w2txt(file_path, data):
    with open(file_path, "w") as file:
        for i in range(data.shape[0]):
            s = ' '.join(str(num) for num in data[i])
            file.write(s+"\n")

if __name__ == '__main__':
    args = parse_args()

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(args.pipeline, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    infer_total_time = 0
    src_tokens_path = os.path.join(args.data_dir, "src_tokens.txt")
    target_sens_path = os.path.join(args.data_dir, "target_sens.txt")
    src_tokens = np.loadtxt(src_tokens_path, dtype=np.int32).reshape(-1, 467)
    target_sens = np.loadtxt(target_sens_path, dtype=np.int32).reshape(-1, 1)
    output_name = "predict_sens.txt"

    stream_name = b'fasttext'
    num = src_tokens.shape[0]
    predictions = []
    for idx in range(num):
        tensor0 = src_tokens[idx]
        tensor0 = np.expand_dims(tensor0, 0)
        tensor_bytes0 = tensor0.tobytes()
        send_source_data(tensor0, tensor_bytes0, stream_name, stream_manager_api, 0)

        tensor1 = np.array([len(src_tokens[idx])]).astype(np.int32)
        tensor1 = np.expand_dims(tensor1, 0)
        tensor_bytes1 = tensor1.tobytes()
        send_source_data(tensor1, tensor_bytes1, stream_name, stream_manager_api, 1)

        # Obtain the inference result by specifying streamName and uniqueId.
        start_time = datetime.datetime.now()
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)
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
        output = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)
        predictions.append(output)

    predictions_sents = np.array(predictions).astype(np.int32)
    w2txt("sdk_predictions_sens.txt", predictions_sents)

    # computer accuracy
    from sklearn.metrics import accuracy_score, classification_report
    target_sens = np.array(target_sens).flatten()
    merge_target_sens = []
    target_label1 = ['0', '1', '2', '3']
    for target_sen in target_sens:
        merge_target_sens.extend([target_sen])
    target_sens = merge_target_sens
    predictions = np.array(predictions).flatten()
    merge_predictions = []
    for prediction in predictions:
        merge_predictions.extend([prediction])
    predictions = merge_predictions
    acc = accuracy_score(target_sens, predictions)

    result_report = classification_report(target_sens, predictions, target_names=target_label1)
    print("********Accuracy: ", acc)
    print(result_report)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
