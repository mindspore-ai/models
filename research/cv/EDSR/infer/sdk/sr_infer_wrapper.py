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
"""super resolution infer wrapper"""
import json
import numpy as np
from PIL import Image
import cv2
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

DEFAULT_IMAGE_WIDTH = 1020
DEFAULT_IMAGE_HEIGHT = 1020
CHANNELS = 3
SCALE = 2

def padding(img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {img.shape}")
    if dh != 0 or dw != 0:
        img = np.pad(img, ((0, int(dh)), (0, int(dw)), (0, 0)), "reflect")
    return img



def unpadding(img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    if img_h > h:
        img = img[:h, :, :]
    if img_w > w:
        img = img[:, :w, :]
    return img



class SRInferWrapper:
    """super resolution infer wrapper"""
    def __init__(self):
        self.stream_name = None
        self.streamManagerApi = StreamManagerApi()
        # init stream manager
        if self.streamManagerApi.InitManager() != 0:
            raise RuntimeError("Failed to init stream manager.")

    def load_pipeline(self, pipeline_path):
        # create streams by pipeline config file
        with open(pipeline_path, 'r') as f:
            pipeline = json.load(f)
        self.stream_name = list(pipeline.keys())[0].encode()
        pipelineStr = json.dumps(pipeline).encode()
        if self.streamManagerApi.CreateMultipleStreams(pipelineStr) != 0:
            raise RuntimeError("Failed to create stream.")

    def do_infer(self, image_path):
        """do infer process"""
        # construct the input of the stream
        image = cv2.imread(image_path)
        ori_h, ori_w, _ = image.shape
        image = padding(image, (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH))
        tensor_pkg_list = MxpiDataType.MxpiTensorPackageList()
        tensor_pkg = tensor_pkg_list.tensorPackageVec.add()
        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0

        for dim in [1, *image.shape]:
            tensor_vec.tensorShape.append(dim)

        input_data = image.tobytes()
        tensor_vec.dataStr = input_data
        tensor_vec.tensorDataSize = len(input_data)

        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = b'appsrc0'
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensor_pkg_list.SerializeToString()
        protobuf_vec.push_back(protobuf)

        unique_id = self.streamManagerApi.SendProtobuf(
            self.stream_name, 0, protobuf_vec)
        if unique_id < 0:
            raise RuntimeError("Failed to send data to stream.")

        # get plugin output data
        key = b"mxpi_tensorinfer0"
        keyVec = StringVector()
        keyVec.push_back(key)
        inferResult = self.streamManagerApi.GetProtobuf(self.stream_name, 0, keyVec)
        if inferResult.size() == 0:
            raise RuntimeError("inferResult is null")
        if inferResult[0].errorCode != 0:
            raise RuntimeError("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                inferResult[0].errorCode, inferResult[0].messageName.decode()))

        # get the infer result
        inferList0 = MxpiDataType.MxpiTensorPackageList()
        inferList0.ParseFromString(inferResult[0].messageBuf)
        inferVisionData = inferList0.tensorPackageVec[0].tensorVec[0].dataStr

        # converting the byte data into 32 bit float array
        output_img_data = np.frombuffer(inferVisionData, dtype=np.float32)
        output_img_data = np.clip(output_img_data, 0, 255)
        output_img_data = np.round(output_img_data).astype(np.uint8)
        output_img_data = np.reshape(output_img_data, (CHANNELS, SCALE*DEFAULT_IMAGE_HEIGHT, SCALE*DEFAULT_IMAGE_WIDTH))
        output_img_data = output_img_data.transpose((1, 2, 0))
        output_img_data = unpadding(output_img_data, (SCALE*ori_h, SCALE*ori_w))
        result = Image.fromarray(output_img_data[..., ::-1])

        return result
