# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

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

import datetime
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/dscnn.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    infer_total_time = 0
    data_path = '../data/input/validation_data.txt'
    label_path = '../data/input/validation_label.txt'

    all_test_data = np.loadtxt(data_path)
    all_test_label = np.loadtxt(label_path)
    all_test_label = all_test_label.astype(np.int32)
    stream_name = b'im_dscnn'
    num = all_test_data.shape[0]
    dataset = np.zeros([num, 1, 49, 20], np.float32)

    for idx in range(num):
        dataset[idx, :, :, :] = all_test_data[idx].reshape(49, 20)
    top1_correct = 0
    top5_correct = 0
    res = []
    for idx in range(num):
        tensor = dataset[idx]
        tensor = np.expand_dims(tensor, 0)
        tensor_bytes = tensor.tobytes()
        in_plugin_id = 0
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
        start_time = datetime.datetime.now()
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
        output = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[-5:]
        res.append(top1_output)
        top1_correct += np.equal(top1_output, all_test_label[idx].astype(np.int32))
        top5_correct += (1 if all_test_label[idx] in top5_output else 0)
    with open('result.txt', 'w') as f:
        for output in res:
            f.write(str(output) + '\n')
    acc1 = 100 * top1_correct / num
    acc5 = 100 * top5_correct / num
    print('Eval: top1_cor:{}, top5_cor:{}, tot:{}, acc@1={:.2f}%, acc@5={:.2f}%' \
                     .format(top1_correct, top5_correct, num, acc1, acc5))

    # destroy streams
    stream_manager_api.DestroyAllStreams()
