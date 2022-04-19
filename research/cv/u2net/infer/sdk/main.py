# coding=utf-8

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
import sys
import time
import MxpiDataType_pb2 as MxpiDataType
import numpy as np

import cv2
import imageio
from PIL import Image
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    Args:
        appsrc_id: an RGB image:the appsrc component number for SendProtobuf
        tensor: the tensor type of the input file
        stream_name: stream Name
        stream_manager:the StreamManagerApi
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


def normPRED(d):
    """rescale the value of tensor to between 0 and 1"""
    ma = d.max()
    mi = d.min()
    dn = (d - mi) / (ma - mi)
    return dn


def normalize(img):
    """normalize tensor"""
    if len(img.shape) == 3:
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    else:
        img = (img - 0.485) / 0.229
    return img


def resize_im(img, size=320):
    """pre process the image"""
    img = img / 255
    img = normalize(img)
    h, w = img.shape[:2]
    img = cv2.resize(img, dsize=(0, 0), fx=size / w, fy=size / h)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2).repeat(1, axis=2)
    im = img
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 0, 1)
    im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
    return im


def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

        # create streams by pipeline config file
    with open("../data/config/u2net.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'u2net_infer'

    # Construct the input of the stream
    # data_input = MxDataInput()
    infer_total_time = 0
    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)  #
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        original = np.array(Image.open(file_path), dtype='float32')
        h, w = original.shape[:2]
        tensor = resize_im(original, size=320)
        # infer
        if not send_source_data(0, tensor, stream_name, stream_manager_api):
            return

        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
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

        # postprocess
        res = res.reshape((1, 7, 320, 320))
        res = normPRED(res[0][0])
        res = cv2.resize(res, dsize=(0, 0), fx=w / res.shape[1], fy=h / res.shape[0])

        # save as image
        save_path = os.path.join(res_dir_name, file_name)
        save_path = save_path.replace(".jpg", ".png")
        imageio.imsave(save_path, res)

    # print the total time of inference
    print("The total time of inference is {} s".format(infer_total_time))

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
