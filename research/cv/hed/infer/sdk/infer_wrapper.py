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
"""super resolution infer"""
import json
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import StringVector
from StreamManagerApi import InProtobufVector
from StreamManagerApi import MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from PIL import Image
import cv2

DEFAULT_IMAGE_WIDTH = 321
DEFAULT_IMAGE_HEIGHT = 481
CHANNELS = 3
SCALE = 2

def get_input(img_file):
    img_file = np.array(cv2.imread(img_file), dtype=np.float32)
    if img_file.shape[0] > img_file.shape[1]:
        img_file = np.rot90(img_file, 1).copy()
    img_file -= np.array((104.00698793, 116.66876762, 122.67891434))
    img_file = np.transpose(img_file, (2, 0, 1))
    print("img_file: ", img_file)
    print("img_file.shape: ", img_file.shape)
    return img_file

class SDKInferWrapper:
    """SDKInferWrapper"""
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
        """do infer stream"""
        # construct the input of the stream
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            image = get_input(image_path)
        elif image_path.endswith(".bin"):
            image = np.fromfile(image_path, dtype=np.float32)
            image.shape = 3, 321, 481
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
        print("len(np from buffer): ", output_img_data.shape)

        result_mat = np.reshape(output_img_data, (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))
        print("result_mat: ", result_mat)
        print("result_mat.shape: ", result_mat.shape)

        result_png = np.round(output_img_data*255).astype(np.uint8)
        result_png = np.reshape(result_png, (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))
        print("result_png.shape: ", result_png.shape)
        result_png = Image.fromarray(result_png)

        return result_png, result_mat
