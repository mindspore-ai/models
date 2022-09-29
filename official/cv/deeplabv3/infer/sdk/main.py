# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import base64
import json
import os
import cv2
import numpy as np
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import InProtobufVector, MxProtobufIn, MxDataInput
import MxpiDataType_pb2 as MxpiDataType
from get_dataset_colormap import label_to_color_image

PIPELINE_PATH = "../data/config/deeplabv3.pipeline"
INFER_RESULT_DIR = "./result"


def _parse_args():
    # val data
    parser = argparse.ArgumentParser('mindspore deeplabv3 eval')
    parser.add_argument('data_root', type=str, default='', help='root path of val data')
    parser.add_argument('data_lst', type=str, default='', help='list of val data')
    parser.add_argument('num_classes', type=int, default=21, help='number of classes')
    args, _ = parser.parse_known_args()
    return args


def _cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def _init_stream(pipeline_path):
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        raise RuntimeError(f"Failed to init stream manager, ret={ret}")

    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()

        ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise RuntimeError(f"Failed to create stream, ret={ret}")
        return stream_manager_api


def _do_infer(stream_manager_api, data_input, origin_img):
    stream_name = b'segmentation'
    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionInfo.format = 0
    vision_vec.visionInfo.width = origin_img.shape[1]
    vision_vec.visionInfo.height = origin_img.shape[0]
    vision_vec.visionInfo.widthAligned = origin_img.shape[1]
    vision_vec.visionInfo.heightAligned = origin_img.shape[0]

    vision_vec.visionData.memType = 0

    vision_vec.visionData.dataStr = data_input.tobytes()
    vision_vec.visionData.dataSize = len(data_input)

    protobuf = MxProtobufIn()
    protobuf.key = b"appsrc0"
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = vision_list.SerializeToString()
    protobuf_vec = InProtobufVector()

    protobuf_vec.push_back(protobuf)
    unique_id = stream_manager_api.SendProtobuf(stream_name, 0, protobuf_vec)

    if unique_id < 0:
        raise RuntimeError("Failed to send data to stream.")

    infer_result = stream_manager_api.GetResult(stream_name, unique_id)

    if infer_result.errorCode != 0:
        raise RuntimeError(
            "GetResultWithUniqueId error, errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))

    load_dict = json.loads(infer_result.data.decode())
    image_mask = load_dict["MxpiImageMask"][0]
    data_str = base64.b64decode(image_mask['dataStr'])
    shape = image_mask['shape']
    return np.frombuffer(data_str, dtype=np.uint8).reshape(shape)


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def pre_process(img_, crop_size=513):
    image_mean = [103.53, 116.28, 123.675]
    image_std = [57.375, 57.120, 58.395]
    # resize
    img_ = resize_long(img_, crop_size)

    # mean, std
    image_mean = np.array(image_mean)
    image_std = np.array(image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    return img_

def main():
    args = _parse_args()
    # init stream manager
    stream_manager_api = _init_stream(PIPELINE_PATH)
    if not stream_manager_api:
        exit(1)

    with open(args.data_lst) as f:
        img_lst = f.readlines()
        os.makedirs(INFER_RESULT_DIR, exist_ok=True)
        data_input = MxDataInput()
        hist = np.zeros((args.num_classes, args.num_classes))
        for _, line in enumerate(img_lst):
            img_path, msk_path = line.strip().split(' ')
            img_path = os.path.join(args.data_root, img_path)
            msk_path = os.path.join(args.data_root, msk_path)
            msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            img_ = cv2.imread(img_path)
            re_img = img_

            crop_size = 513
            img_ = pre_process(img_, crop_size)

            data_input = img_.astype("float32")

            each_array = _do_infer(stream_manager_api, data_input, re_img)

            hist += _cal_hist(
                msk_.flatten(), each_array.flatten(), args.num_classes)
            color_mask_res = label_to_color_image(each_array)
            color_mask_res = cv2.cvtColor(color_mask_res.astype(np.uint8),
                                          cv2.COLOR_RGBA2BGR)
            result_path = os.path.join(
                INFER_RESULT_DIR,
                f"{img_path.split('/')[-1].split('.')[0]}.png")
            cv2.imwrite(result_path, color_mask_res)
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("per-class IoU", iou)
        print("mean IoU", np.nanmean(iou))

    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    main()
