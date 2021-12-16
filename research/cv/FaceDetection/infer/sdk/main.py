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
"""sdk infer"""
import sys

import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn

from util.data_preprocess import SingleScaleTrans
from util.eval import gen_eval_result, eval_according_output
from util.eval_util import prepare_file_paths, get_data


def send_appsrc_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor = np.expand_dims(tensor, 0)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
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


def run_pipeline(dataset_path):
    """
        enable comparison graph output
        :param: the path of data_set
        :returns: null
        Output: the figure of accuracy
    """
    # Init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Create streams by pipeline pipeline config file
    with open("pipeline/faceDetection.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # record the eval times
    eval_times = 0
    # record the infer result
    det = {}
    # record the image size
    img_size = {}
    # record the image label
    img_anno = {}

    # loop through the image, start to output the results, and evaluate the accuracy
    print('=============FaceDetection start evaluating==================')
    image_files, anno_files, image_names = prepare_file_paths(dataset_path)
    dataset_size = len(anno_files)
    assert dataset_size == len(image_files)
    assert dataset_size == len(image_names)
    data_set = []
    for i in range(dataset_size):
        data_set.append(get_data(image_files[i], anno_files[i], image_names[i]))
    for data in data_set:
        input_shape = [768, 448]
        single_trans = SingleScaleTrans(resize=input_shape)
        pre_data = single_trans.__call__(data['image'], data['annotation'], data['image_name'], data['image_size'])
        images, labels, image_name, image_size = pre_data[0:4]
        # print(labels)
        # exit()
        # Judge the input picture whether is a jpg format
        try:
            if data['image'][0:4] != b'\xff\xd8\xff\xe0' or data['image'][6:10] != b'JFIF':
                print('The input image is not the jpg format.')
                exit()
        except IOError:
            print('an IOError occurred while opening {}, maybe your input is not a picture'.format(image_names))
            exit()

        # Inputs convert to a specified stream based on stream_name.
        stream_name = b'faceDetection'
        in_plugin_id = 0
        if not send_appsrc_data(in_plugin_id, images[0], stream_name, stream_manager_api):
            return
        # Get result numpy array
        keys = [b"mxpi_tensorinfer0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)

        # Get inference result
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % infer_result[0].errorCode)
            return

        # Get the result of the model
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)

        # take 6 outcomes from the matrix: coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2
        coords_0 = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32) \
            .reshape(1, 4, 84, 4)
        cls_scores_0 = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32) \
            .reshape(1, 4, 84)
        coords_1 = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32) \
            .reshape(1, 4, 336, 4)
        cls_scores_1 = np.frombuffer(result.tensorPackageVec[0].tensorVec[3].dataStr, dtype=np.float32) \
            .reshape(1, 4, 336)
        coords_2 = np.frombuffer(result.tensorPackageVec[0].tensorVec[4].dataStr, dtype=np.float32) \
            .reshape(1, 4, 1344, 4)
        cls_scores_2 = np.frombuffer(result.tensorPackageVec[0].tensorVec[5].dataStr, dtype=np.float32) \
            .reshape(1, 4, 1344)

        eval_according_output(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2, det,
                              img_anno, img_size, labels, image_name, image_size)
        eval_times += 1

    # generate the eval result
    gen_eval_result(eval_times, det, img_size, img_anno)
    # Destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    arg_arr_input_output_path = []
    if len(sys.argv) != 2:
        print('Wrong parameter setting.')
        exit()

    run_pipeline(sys.argv[1])
