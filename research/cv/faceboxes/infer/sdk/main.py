
"""
# Copyright(C) 2022. Huawei Technologies Co.,Ltd.
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
"""

import os
import argparse
from itertools import product
from math import ceil
import cv2
import numpy as np
import torch
import torch.nn as nn
from eval_result import DetectionEngine

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


parser = argparse.ArgumentParser(description='FaceBoxes: Face Detection')
parser.add_argument('--image-path', type=str, default="../data/input", help='input of images path')
args_opt = parser.parse_args()

def decode_bbox(bbox, priors_decode, var):
    """decode_bbox"""
    half = 2
    boxes_decode = np.concatenate((
        priors_decode[:, 0:2] + bbox[:, 0:2] * var[0] * priors_decode[:, 2:4],
        priors_decode[:, 2:4] * np.exp(bbox[:, 2:4] * var[1])), axis=1)
    boxes_decode[:, :2] -= boxes_decode[:, 2:] / half
    boxes_decode[:, 2:] += boxes_decode[:, :2]
    return boxes_decode


def _nms(boxes_nms, threshold=0.5):
    """nms"""
    x1 = boxes_nms[:, 0]
    y1 = boxes_nms[:, 1]
    x2 = boxes_nms[:, 2]
    y2 = boxes_nms[:, 3]
    scores = boxes_nms[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    reserved_boxes = []
    while order.size > 0:
        ii = order[0]
        reserved_boxes.append(ii)
        max_x1 = np.maximum(x1[ii], x1[order[1:]])
        max_y1 = np.maximum(y1[ii], y1[order[1:]])
        min_x2 = np.minimum(x2[ii], x2[order[1:]])
        min_y2 = np.minimum(y2[ii], y2[order[1:]])

        intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
        intersect_area = intersect_w * intersect_h
        ovr = intersect_area / (areas[ii] + areas[order[1:]] - intersect_area)

        indices = np.where(ovr <= threshold)[0]
        order = order[indices + 1]
    print("----reserved_boxes is: ", reserved_boxes, len(reserved_boxes))
    return reserved_boxes


def prior_box(image_size, min_sizes, steps, clip=False):
    """prior box"""
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)]
        for step in steps]

    anchors = []
    for k, ff in enumerate(feature_maps):
        for i_p, j in product(range(ff[0]), range(ff[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                if min_size == 32:
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i_p + 0, i_p + 0.25, i_p + 0.5, i_p + 0.75]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                elif min_size == 64:
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0, j + 0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i_p + 0, i_p + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                else:
                    cx = (j + 0.5) * steps[k] / image_size[1]
                    cy = (i_p + 0.5) * steps[k] / image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4]).astype(np.float32)
    if clip:  # false
        output = np.clip(output, 0, 1)
    return output


def detect(boxes_detect, confs_detect, resize, scale_detect, image_path_de, priors_detect):
    """detect"""
    if boxes_detect.shape[0] == 0:
        # add to result
        event_name, img_name_de = image_path_de.split('/')
        results[event_name][img_name_de[:-4]] = {'img_path': image_path_de,
                                                 'bboxes': []}
        return

    boxes_detect = decode_bbox(np.squeeze(boxes_detect, 0), priors_detect, [0.1, 0.2])
    boxes_detect = boxes_detect * scale_detect / resize

    scores = np.squeeze(confs_detect.numpy(), 0)[:, 1]
    # ignore low scores
    inds = np.where(scores > 0.05)[0]
    boxes_detect = boxes_detect[inds]
    scores = scores[inds]


    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes_detect = boxes_detect[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes_detect, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = _nms(dets, 0.4)
    dets = dets[keep, :]

    dets[:, 2:4] = (dets[:, 2:4].astype(np.int) - dets[:, 0:2].astype(np.int)).astype(np.float)  # int
    dets[:, 0:4] = dets[:, 0:4].astype(np.int).astype(np.float)  # int

    # add to result
    event_name, img_name_de = image_path_de.split('/')
    if event_name not in results.keys():
        results[event_name] = {}
    results[event_name][img_name_de[:-4]] = {'img_path': image_path_de,
                                             'bboxes': dets[:, :5].astype(np.float).tolist()}
    print("----- dets bboxes is", dets[:, :5].astype(np.float), dets[:, :5].astype(np.float).shape)


if __name__ == '__main__':

    test_dataset = []
    with open(os.path.join("../data/input", 'val_img_list.txt'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        test_dataset.append(line.rstrip())

    results = {}

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open("../data/config/faceboxes.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()


    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(args_opt.image_path, 'images', img_name)


        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_t = np.empty((2496, 1056, 3), dtype=img.dtype)
        image_t[:, :] = (104, 117, 123)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
        if not os.path.exists("../data/input/imagespadding/" + img_name.split("/")[0]):
            os.makedirs("../data/input/imagespadding/" + img_name.split("/")[0])
        cv2.imwrite("../data/input/imagespadding/" + img_name, img)
        image_path = os.path.join("../data/input", 'imagespadding', img_name)

        dataInput = MxDataInput()

        if os.path.exists(image_path) != 1:
            print("The test image does not exist.")

        try:
            with open(image_path, 'rb') as f:
                dataInput.data = f.read()
        except FileNotFoundError:
            print("Test image", "test.jpg", "doesn't exist. Exit.")
            exit()
        streamName = b'detection'
        inPluginId = 0
        uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        keyVec = StringVector()
        keyVec.push_back(b"mxpi_tensorinfer0")
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()

        YUV_BYTES_NU = 3
        YUV_BYTES_DE = 2

        print("infer_result size: ", len(infer_result))

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)

        print("result.tensorPackageVec size: ", len(result.tensorPackageVec))
        print("result.tensorPackageVec[0].tensorVec size: ",
              len(result.tensorPackageVec[0].tensorVec))

        print("len tensor0", len(np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)))
        print("len tensor1", len(np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)))
        boxes = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(1, 54897, 4)
        confs = torch.from_numpy(
            np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32).reshape(1, 54897, 2))
        confs_sofmax = nn.Softmax(dim=2)
        confs = confs_sofmax(confs)
        print("------img_name is", img_name)

        scale = np.array([1056, 2496, 1056, 2496], dtype=np.float32)
        priors = prior_box(image_size=(2496, 1056),
                           min_sizes=[[32, 64, 128], [256], [512]],
                           steps=[32, 64, 128], clip=False)
        detect(boxes, confs, 1, scale, img_name, priors)

    print('============== Eval starting ==============')
    detection = DetectionEngine(results)
    detection.get_eval_result()
    print('============== Eval done ==============')
    # destroy streams
    streamManagerApi.DestroyAllStreams()
