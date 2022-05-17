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
import glob
from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import cv2


shape = [1, 3, 640, 640]


if __name__ == '__main__':
    pipeline_path = "../data/config/nasfpn.pipeline"
    data_path = "../../coco/val2017"
    result_path = "result"
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream


    if not os.path.exists(result_path):
        os.makedirs(result_path)



    file_list = sorted(glob.glob(data_path+"/*"))
    img_size = len(file_list)
    results = []

    for idx, file in enumerate(file_list[:]):
        file_id = file.replace('.jpg', '').split('/')[-1]
        print(file_id)
        data = cv2.imread(file, cv2.IMREAD_COLOR)
        data = cv2.resize(data, (640, 640))[:, :, [2, 1, 0]]
        means, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
        data = (data - means)/ std
        data = data.transpose((2, 0, 1))[None, :].astype(np.float32)
        # Construct the input of the stream
        data_input = MxDataInput()
        data_input.data = data.tobytes()
        tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
        tensorVec1 = tensorPackage1.tensorVec.add()
        tensorVec1.deviceId = 0
        tensorVec1.memType = 0
        for t in data.shape:
            tensorVec1.tensorShape.append(t)
        tensorVec1.dataStr = data_input.data
        tensorVec1.tensorDataSize = len(data.tobytes())
        protobufVec1 = InProtobufVector()
        protobuf1 = MxProtobufIn()
        protobuf1.key = b'appsrc0'
        protobuf1.type = b'MxTools.MxpiTensorPackageList'
        protobuf1.protobuf = tensorPackageList1.SerializeToString()
        protobufVec1.push_back(protobuf1)

        unique_id = stream_manager.SendProtobuf(b'nasfpn', b'appsrc0', protobufVec1)

        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'nasfpn', 0, keyVec)
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

        boxes = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(76725, 4)
        score = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32).reshape(76725, 81)

        boxes.tofile(os.path.join(result_path, file_id + '_0.bin'))
        score.tofile(os.path.join(result_path, file_id + '_1.bin'))

    # destroy streams
    stream_manager.DestroyAllStreams()
