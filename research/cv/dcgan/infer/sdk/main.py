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
""" Model Main """

import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector

NORMALIZE_MEAN = 127.5
NORMALIZE_STD = 127.5
PLT_SHAPE = [4, 4]

def create_input():
    """
    Create an random input tensor, size is (16, 100, 1, 1).

    Args:
        None

    Returns:
        input_data_(numpy.ndarray): An random tensor, size is (16, 100, 1, 1).

    """
    input_data_ = np.random.normal(size=(16, 100, 1, 1)).astype("float32")

    return input_data_


def postprocess_image(result_np_):
    """
    Do postprocess to the result of inference, include denormalize and convert from NCHW to NHWC.

    Args:
        result_np_(numpy.ndarray): The result of inference, size is (3, 32, 32).

    Returns:
        result_np_(numpy.ndarray): Postprocessed result, size is (32, 32, 3).
    """
    # denormalize
    result_np_ = (result_np_ * NORMALIZE_STD + NORMALIZE_MEAN).astype(np.uint8)
    # NCHW to NHWC
    result_np_ = result_np_.transpose((1, 2, 0))

    return result_np_


def save_images(result_np_, save_path_):
    """
    Save results of inference to images in groups of 16 (which depends on the model shape)

    Args:
        result_np_(numpy.ndarray): The result of inference, size is (16, 3, 32, 32).
        save_path_(str): File path to save images in.

    Returns:
        None

    """
    result_ = result_np_.copy()

    for ii in range(result_.shape[0]):
        image = postprocess_image(result_[ii])
        plt.subplot(PLT_SHAPE[0], PLT_SHAPE[1], ii + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.savefig(save_path_)


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("[INFO] Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    print("[INFO] Init Stream manager successfully!")

    # create streams by pipeline config file
    with open("../data/config/DCGAN.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("[INFO] Failed to create Stream, ret=%s" % str(ret))
        exit()
    print("[INFO] Create Stream successfully!")

    res_dir_name = sys.argv[1]
    n_epochs = int(sys.argv[2])

    np.random.seed(1213)
    for epoch in range(n_epochs):
        print("[INFO] Epoch:", epoch)

        # Get input data
        input_data = create_input()

        # Construct the input of the stream
        mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()

        tensorVec = tensor_package_vec.tensorVec.add()
        tensorVec.memType = 1
        tensorVec.deviceId = 0

        tensorVec.tensorDataSize = int(
            input_data.shape[0]*input_data.shape[1]*input_data.shape[2]*input_data.shape[3]*4)
        tensorVec.tensorDataType = 0  # float32
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
        stream_name = b'DCGAN'
        in_plugin_id = 0

        # Send data to stream
        unique_id = stream_manager_api.SendProtobuf(
            stream_name, in_plugin_id, protobuf_vec)
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

        # Save result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)

        result_np = np.frombuffer(
            result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        result_np = result_np.reshape(16, 3, 32, 32)

        save_path = res_dir_name+"/"+str(epoch)+".jpg"
        save_images(result_np, save_path)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
