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

import datetime
import os
import sys
import numpy as np
import cv2
import MxpiDataType_pb2 as MxpiDataType

from PIL import Image
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import StringVector, MxProtobufIn, InProtobufVector


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            ow = int(size * w / h)
            oh = size
        return img.resize((ow, oh), interpolation)

    return img.resize(size[::-1], interpolation)


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
    with open("../data/config/NIMA.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    # Construct the input of the stream

    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(
                ".jpg") or file_name.lower().endswith(".jpeg")):
            return

        # image preprocess
        img_cv = cv2.imread(file_path)[:, :, ::-1]
        img_cv = cv2.resize(img_cv, (224, 224), interpolation=cv2.INTER_LINEAR)
        print(img_cv.shape)
        # normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean)
        std = np.array(std)
        img_np = np.array(img_cv)
        img_np = (img_np - mean) / std
        # transpose
        img_np = img_np.transpose((2, 0, 1))
        img_np = img_np.astype(np.float32)

        stream_name = b'im_nima'
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 0
        vision_vec.visionInfo.width = 224
        vision_vec.visionInfo.height = 224
        vision_vec.visionInfo.widthAligned = 224
        vision_vec.visionInfo.heightAligned = 224

        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataType = 1
        vision_vec.visionData.dataStr = img_np.tobytes()
        vision_vec.visionData.dataSize = len(img_np)

        protobuf = MxProtobufIn()
        protobuf.key = b"appsrc0"
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec = InProtobufVector()

        protobuf_vec.push_back(protobuf)

        # Inputs data to a specified stream based on streamName.
        inplugin_id = 0

        # Send data to stream
        unique_id = stream_manager_api.SendProtobuf(stream_name, inplugin_id, protobuf_vec)
        if unique_id < 0:
            print("Failed to send data to stream.")
            return

        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetProtobuf(stream_name, unique_id, keyVec)
        print(infer_result)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.size() == 0:
            print("[INFO] infer result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            return

        result = MxpiDataType.MxpiTensorPackageList()

        # Get the first result(only 1 input image)
        result.ParseFromString(infer_result[0].messageBuf)
        result_np = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        print("result_np_size:", result_np.size)
        print("process img: {}, result_np: {}".format(file_name, result_np))

        str_array = " ".join(map(str, result_np))
        image_name = file_name + ":" + str_array
        print("image_name:", image_name)
        with open(res_dir_name + "/test.txt", 'a+') as f_write:
            f_write.writelines(image_name)
            f_write.write('\n')

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
