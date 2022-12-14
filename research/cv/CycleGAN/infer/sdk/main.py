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
import datetime
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn
import cv2


def process_img(img_file):
    """ Preprocess Image """
    # read image
    img_cv = cv2.imread(img_file)
    img_shape_ = img_cv.shape
    print("[INFO] File:", img_file, " ImageSize:", img_shape_)
    # resize to 256*256
    img_cv = cv2.resize(img_cv, (256, 256))
    # BGR to RGB
    img_np_ = np.array(img_cv)
    # NHWC to NCHW (opencv format -> model inference format)
    img_np_ = img_np_.transpose((2, 0, 1))
    # normalize
    img_np_ = np.array((img_np_-127.5)/127.5).astype(np.float32)
    # expend dims for inference
    img_np_ = np.expand_dims(img_np_, axis=0)

    return img_np_, img_shape_


def postprocess_image(result_np_, img_shape_):
    """ Decode Image """
    # reduce dims: already done

    # denormalize
    result_np_ = (result_np_ * 127.5 + 127.5).astype(np.uint8)
    # NCHW to NHWC
    result_np_ = result_np_.transpose((1, 2, 0))
    result_cv_ = cv2.resize(result_np_, (img_shape_[1], img_shape_[0]))

    return result_cv_


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("[INFO] Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    print("[INFO] Init Stream manager successfully!")

    # create streams by pipeline config file
    with open("CycleGAN.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("[INFO] Failed to create Stream, ret=%s" % str(ret))
        exit()
    print("[INFO] Create Stream successfully!")

    # Get file path
    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for epoch, file_name in enumerate(file_list):
        print("[INFO] epoch:", epoch)

        # load file
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".jpg")
                or file_name.lower().endswith(".jpeg")):
            continue

        # image processing
        img_np, img_shape = process_img(file_path)
        input_data = img_np

        # Construct the input of the stream
        mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()

        tensorVec = tensor_package_vec.tensorVec.add()
        tensorVec.memType = 1
        tensorVec.deviceId = 0

        tensorVec.tensorDataSize = int(
            input_data.shape[0]*input_data.shape[1]*input_data.shape[2]*input_data.shape[3]*4)
        tensorVec.tensorDataType = 0
        for i in input_data.shape:
            tensorVec.tensorShape.append(i)
        tensorVec.dataStr = input_data.tobytes()

        protobuf = MxProtobufIn()
        protobuf.key = b'appsrc0'
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = mxpi_tensor_package_list.SerializeToString()

        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)

        # Inputs data to a specified stream based on streamName.
        stream_name = b'CycleGAN'
        inplugin_id = 0

        # Send data to stream
        unique_id = stream_manager_api.SendProtobuf(
            stream_name, inplugin_id, protobuf_vec)
        if unique_id < 0:
            print("[INFO] Failed to send data to stream.")
            exit()
        print("[INFO] Send data to stream successfully!")

        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetProtobuf(
            stream_name, unique_id, keyVec)
        end_time = datetime.datetime.now()
        print('[INFO] sdk run time: {}'.format(
            (end_time - start_time).microseconds))
        if infer_result.size() == 0:
            print("[INFO] infer result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("[INFO] GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" %
                  (infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()
        print("[INFO] Get result successfully!")

        # Transform result from buffer to numpy
        result = MxpiDataType.MxpiTensorPackageList()

        # Get the first result(only 1 input image)
        result.ParseFromString(infer_result[0].messageBuf)
        result_np = np.frombuffer(
            result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        result_np = result_np.reshape(3, 256, 256)

        # result reprocessing
        result_cv = postprocess_image(result_np, img_shape)

        # save result
        save_path = res_dir_name+"/"+file_name
        cv2.imwrite(save_path, result_cv)
        print("[INFO] Result saved to:", save_path)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
