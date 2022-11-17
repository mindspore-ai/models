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
import os
import operator
from functools import reduce
import math
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput
from dataset import testdataset
import numpy as np
import cv2
from pypse import pse


def sort_to_clockwise(points_):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points_), [len(points_)] * 2))
    clockwise_points = sorted(points_, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    return clockwise_points


def write_result_as_txt(img_name_, bboxes_, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = os.path.join(path, 'res_{}.txt'.format(os.path.splitext(img_name_)[0]))
    lines = []
    for _, bbox_ in enumerate(bboxes_):
        bbox_ = bbox_.reshape(-1, 2)
        bbox_ = np.array(list(sort_to_clockwise(bbox_)))[[3, 0, 1, 2]].copy().reshape(-1)
        values = [int(v) for v in bbox_]
        line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
        lines.append(line)
    with open(filename, 'w') as ff:
        for line in lines:
            ff.write(line)


if __name__ == '__main__':

    local_path = ""
    if not os.path.isdir('{}./res/submit_ic15/'.format(local_path)):
        os.makedirs('{}./res/submit_ic15/'.format(local_path))
    if not os.path.isdir('{}./res/vis_ic15/'.format(local_path)):
        os.makedirs('{}./res/vis_ic15/'.format(local_path))

    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("../data/pipline/psenet.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    ds = testdataset(test_root_dir='../data/image', resize=1920)

    for data in ds:
        # get data
        img, img_resized, img_name = data
        print(img_name)
        img = img.astype(np.uint8).copy()
        img_resized = img_resized
        cv2.imwrite('1.jpg', img_resized)

        with open('1.jpg', 'rb') as f:
            dataInput.data = f.read()

        # Inputs data to a specified stream based on streamName.
        streamName = b'psenet'
        inPluginId = 0

        uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId.
        keys = [b"mxpi_tensorinfer0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)
        infer_result = streamManagerApi.GetProtobuf(streamName, inPluginId, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("infer_result error. errorCode=%d" % (infer_result[0].errorCode))
            exit()
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res1 = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        res2 = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)
        shape = (1920, 1920)
        score = res1.reshape(shape)
        kernels = res2.reshape((7, *shape)).astype(np.uint8)
        score = np.squeeze(score)
        kernels = np.squeeze(kernels)
        pred = pse(kernels, 5.0)

        scale = max(img.shape[:2]) * 1.0 / 1920
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < 600:
                continue

            score_i = np.mean(score[label == i])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            cv2.drawContours(img, [bbox], 0, (0, 255, 0), 3)
            bboxes.append(bbox)

        cv2.imwrite('{}./res/vis_ic15/{}'.format(local_path, img_name), img[:, :, [2, 1, 0]].copy())
        write_result_as_txt(img_name, bboxes, '{}./res/submit_ic15/'.format(local_path))

    streamManagerApi.DestroyAllStreams()
