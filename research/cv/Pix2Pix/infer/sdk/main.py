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
import os
import numpy as np
import cv2

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput
from StreamManagerApi import StringVector
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import InProtobufVector

def process_img(img_file):
    """ Preprocess Image, get the input_img(right_side) """

    AB = cv2.imread(img_file)   # HWC

    # h = AB.shape[0]
    w = AB.shape[1]
    w2 = int(w / 2)

    B = AB[:, w2:w]

    img_B = cv2.resize(B, (256, 256))  # resize
    img_B = np.array((img_B - 127.5) / 127.5).astype(np.float32)  # Normalization
    img_B = img_B[:, :, ::-1].transpose((2, 0, 1))  # HWC2CHW

    return img_B

def decode_image(imgAfterInfer):
    """ Decode Image """
    mean = 0.5 * 255
    std = 0.5 * 255
    return (imgAfterInfer * std + mean).astype(np.uint8).transpose((1, 2, 0))


if __name__ == '__main__':

    img_dir = "../data/test_img"
    result_dir = "../data/sdk_result"
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./Pix2Pix.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    all_start_time = datetime.datetime.now()
    file_list = os.listdir(img_dir)
    for _, file_name in enumerate(file_list):
        start_time = datetime.datetime.now()
        print(file_name)
        file_path = os.path.join(img_dir, file_name)
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        print(file_path)

        empty_data = []
        stream_name = b'Pix2Pix'
        in_plugin_id = 0
        input_key = 'appsrc0'


        img_np = process_img(file_path)                  # (3, 256, 256)
        img_np = np.expand_dims(img_np, axis=0)          # (1, 3, 256, 256)
        print("***************************************************************************************")
        print(img_np.shape)
        print("***************************************************************************************")

        tensor_list = MxpiDataType.MxpiTensorPackageList()
        tensor_pkg = tensor_list.tensorPackageVec.add()

        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(img_np.shape)
        tensor_vec.tensorDataType = 0
        tensor_vec.dataStr = img_np.tobytes()
        tensor_vec.tensorDataSize = len(img_np)
        buf_type = b"MxTools.MxpiTensorPackageList"

        protobuf = MxProtobufIn()
        protobuf.key = input_key.encode("utf-8")
        protobuf.type = buf_type
        protobuf.protobuf = tensor_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)
        err_code = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
        if err_code != 0:
            print(
                "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
                "buf_type(%s), err_code(%s).", stream_name, in_plugin_id,
                input_key, buf_type, err_code)


        keys = [b"mxpi_tensorinfer0",]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        TensorList = MxpiDataType.MxpiTensorPackageList()
        TensorList.ParseFromString(infer_result[0].messageBuf)
        data = np.frombuffer(TensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

        end_time = datetime.datetime.now()
        infer_time = end_time - start_time
        print("***************************************************************************************")
        print(f"The image inference time is {infer_time}")
        print("infer finish")
        print("***************************************************************************************")
        data = data.reshape(3, 256, 256)
        img = decode_image(data)
        img = img[:, :, ::-1]
        res_path = result_dir + '/' + file_name
        # print(res_path)
        cv2.imwrite(res_path, img)
        print("***************************************************************************************")
        print("{} saved".format(res_path))
        print("***************************************************************************************")

    all_end_time = datetime.datetime.now()
    all_infer_time = all_end_time - all_start_time
    print("***************************************************************************************")
    print(f"The all image inference time is {all_infer_time}")
    print("All infer finish")
    print("***************************************************************************************")
    # destroy streams
    stream_manager_api.DestroyAllStreams()
