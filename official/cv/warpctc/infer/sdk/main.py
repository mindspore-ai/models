#!/usr/bin/env python

# coding=utf-8

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

import argparse
import os
import glob
import numpy as np
from PIL import Image

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="WarpCTC process")
    parser.add_argument("--pipeline", type=str, default=None, help="SDK infer pipeline")
    parser.add_argument("--image_path", type=str, default=None, help="root path of image without noise")
    parser.add_argument('--image_width', default=160, type=int, help='resized image width')
    parser.add_argument('--image_height', default=64, type=int, help='resized image height')
    parser.add_argument('--channel', default=3, type=int
                        , help='image channel, 3 for color, 1 for gray')
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
    for k in tensor.shape:
        tensor_vec.tensorShape.append(k)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    rete = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if rete < 0:
        print("Failed to send data to stream.")
        return False
    return True


if __name__ == '__main__':
    args = parse_args()
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open(args.pipeline, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    streamName = b'Identify'
    inPluginId = 0
    image_list = glob.glob(os.path.join(args.image_path, '*'))

    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    for image in image_list:
        if os.path.exists(image) != 1:
            print("The test image does not exist.")
        filename = image
        mean = [0.9010, 0.9049, 0.9025]
        std = [0.1521, 0.1347, 0.1458]
        img_data_ori = Image.open(filename)
        img_data_norm = np.array((np.array(img_data_ori, dtype='float16') / 255.0 - mean) / std, dtype='float16')
        img_data = np.expand_dims(img_data_norm.transpose((0, 1, 2)), 0)
        if not send_source_data(0, img_data, streamName, streamManagerApi):
            exit()
        # get inference result
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            exit()
        # print the infer result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        infer_result_ndArray = []
        infer_result_ndArray.append(
            np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16).reshape(-1, 1, 11))

        seq_len, batch_size, _ = infer_result_ndArray[0].shape
        indices = infer_result_ndArray[0].argmax(axis=2)
        lens = [seq_len] * batch_size
        pred_lbl = []
        for i in range(batch_size):
            idx = indices[:, i]
            last_idx = 10
            pred_lbl = []
            for j in range(lens[i]):
                cur_idx = idx[j]
                if cur_idx not in [last_idx, 10]:
                    pred_lbl.append(cur_idx)
                last_idx = cur_idx

        filename = image.split('/')[-1].split('.')[0]  # get the name of image file
        with open(os.path.join("./outputs", filename + '.txt'), "w") as f:
            f.write(str(pred_lbl))

    # destroy streams
    streamManagerApi.DestroyAllStreams()
