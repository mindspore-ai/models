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
""" main.py """
import os
import argparse
from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

import numpy as np
from PIL import Image
import cv2


def parse_args(parser_):
    """
    Parse commandline arguments.
    """
    parser_.add_argument('--images_txt_path', type=str, default="../data/image/images.txt",
                         help='image text')
    parser_.add_argument('--labels_txt_path', type=str, default="../data/image/labels.txt",
                         help='label')
    return parser_


def read_file_list(input_file):
    """
    :param infer file content:
        0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
        1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
        ...
    :return image path list
    """
    image_file_list = []
    if not os.path.exists(input_file):
        print('input file does not exists.')
    with open(input_file, "r") as fs:
        for line in fs.readlines():
            if len(line) > 10:
                line = line.strip('\n').split('\t')[0].replace('\\', '/')
                image_file_list.append(line)
    return image_file_list


image_height = 64
image_width = 512


def resize_image(img):
    color_fill = 255
    scale = image_height / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    if img.shape[1] > image_width:
        img = img[:, 0:image_width]
    else:
        blank_img = np.zeros((image_height, image_width, 3), np.uint8)
        # fill the image with white
        blank_img.fill(color_fill)
        blank_img[:image_height, :img.shape[1]] = img
        img = blank_img
    data = np.array([img[...]], np.float32)
    data = data / 127.5 - 1  # [1,3,64,512]  0-255
    return data.transpose((0, 3, 1, 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Om CNN Direction Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/cnndirection.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream

    res_dir_name = 'result'
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    acc = 0
    acc_1 = 0
    acc_0 = 0
    num_1 = 0
    num_0 = 0
    infer_file = '../data/image/annotation_test.txt'
    file_list = read_file_list(infer_file)

    img_size = len(file_list)
    results = []

    for file in file_list:
        image = Image.open(os.path.join('../data/image/', file))
        image = np.array(image)
        image = resize_image(image)
        label = int(file.split('_')[-1].split('.')[0])

        # Construct the input of the stream
        data_input1 = MxDataInput()
        data_input1.data = image.tobytes()
        tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
        tensorVec1 = tensorPackage1.tensorVec.add()
        tensorVec1.deviceId = 0
        tensorVec1.memType = 0
        for t in image.shape:
            tensorVec1.tensorShape.append(t)
        tensorVec1.dataStr = data_input1.data
        tensorVec1.tensorDataSize = len(image.tobytes())
        protobufVec1 = InProtobufVector()
        protobuf1 = MxProtobufIn()
        protobuf1.key = b'appsrc0'
        protobuf1.type = b'MxTools.MxpiTensorPackageList'
        protobuf1.protobuf = tensorPackageList1.SerializeToString()
        protobufVec1.push_back(protobuf1)

        unique_id = stream_manager.SendProtobuf(b'cnn_direction', b'appsrc0', protobufVec1)

        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager.GetProtobuf(b'cnn_direction', 0, keyVec)

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
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

        results.append(res)

        if label == 0:
            num_0 += 1
            if res[0] > res[1]:
                acc += 1
                acc_0 += 1
        else:
            num_1 += 1
            if res[0] <= res[1]:
                acc += 1
                acc_1 += 1

    results = np.vstack(results)
    np.savetxt("./result/infer_results.txt", results, fmt='%.06f')

    # destroy streams
    stream_manager.DestroyAllStreams()
    print('Eval size:', img_size)
    print('num of label 1:', num_1, 'total acc1:', acc_1 / num_1)
    print('num of label 0:', num_0, 'total acc0:', acc_0 / num_0)
    print('total acc:', acc / img_size)

    with open("../results/eval_sdk.log", 'w') as f:
        f.write('Eval size: {} \n'.format(img_size))
        f.write('num of label 1: {}, total acc1: {} \n'.format(num_1, acc_1 / num_1))
        f.write('num of label 0: {}, total acc0: {} \n'.format(num_0, acc_0 / num_0))
        f.write('total acc: {} \n'.format(acc / img_size))
