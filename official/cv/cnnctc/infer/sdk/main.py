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

import math
import os
import sys
import json
import time

import numpy as np
import cv2

from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput


def preprocess(src_img):
    imgH = 32
    imgW = 100
    max_size = (3, imgH, imgW)
    w, h = src_img.shape[1], img.shape[0]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)
    resized_img = cv2.resize(src_img, (resized_w, imgH), cv2.INTER_CUBIC)
    # toTensor
    resized_img = np.array(resized_img, dtype=np.uint8)
    resized_img = resized_img.transpose([2, 0, 1])

    _, _, w = resized_img.shape

    Pad_img = np.zeros(shape=max_size, dtype=np.uint8)
    Pad_img[:, :, :w] = resized_img  # right pad
    if max_size[2] != w:  # add border Pad
        Pad_img[:, :, w:] = np.tile(np.expand_dims(resized_img[:, :, w - 1], 2), (1, 1, max_size[2] - w))

    Pad_img = Pad_img.transpose([1, 2, 0])
    return Pad_img


if __name__ == '__main__':
    resultPath = "/cnnctc_sdk_result.txt"
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("../data/config/cnnctc.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]

    file_list = os.listdir(dir_name)
    file_list.sort()
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    if os.path.exists(res_dir_name + resultPath):
        os.remove(res_dir_name + resultPath)

    infer_total_time = 0
    infer_total_count = 0
    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        infer_total_count = infer_total_count+1
        img = cv2.imread(file_path)

        newImg = preprocess(img)
        cv2.imwrite('tmp.jpg', newImg)
        with open('tmp.jpg', 'rb') as f:
            data_input.data = f.read()

        stream_name = b'im_cnnctc'
        in_plugin_id = 0
        unique_id = streamManagerApi.SendData(
            stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.

        keys = [b"mxpi_tensorinfer0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)

        start_time = time.time()
        infer_result = streamManagerApi.GetResult(
            stream_name, unique_id)
        infer_total_time += time.time() - start_time
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, "
                  "errorMsg=%s" % (infer_result.errorCode,
                                   infer_result.data.decode()))
            exit()
        # print the infer result
        preds_str = json.loads(infer_result.data.decode())['MxpiTextsInfo'][0]['text']
        print("Prediction samples: \n", preds_str)

        with open(res_dir_name + resultPath, 'a') as f_write:
            f_write.writelines(preds_str)
            f_write.write('\n')

    # destroy streams
    streamManagerApi.DestroyAllStreams()

    print('<<========  Infer Metric ========>>')
    print("Number of samples:%d" % infer_total_count)
    print("Infer total time:%f" % infer_total_time)
    if infer_total_count != 0:
        print("Average infer time:%f" % (infer_total_time / infer_total_count))
        print("Infer count per second:%f" % (infer_total_count / infer_total_time))
    print('<<===============================>>')
