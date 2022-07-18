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
import os
import time
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

INFER_RESULT_DIR = "./result"
IMAGE_MEAN = [103.53, 116.28, 123.675]
IMAGE_STD = [57.375, 57.120, 58.395]

def _parse_args():
    parser = argparse.ArgumentParser('refinenet eval')

    # val data
    parser.add_argument('--data_root', type=str, default='', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--pipeline', type=str, default='', help='pipeline file path')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
    parser.add_argument('--image_width', default=513, type=int, help='image width')
    parser.add_argument('--image_height', default=513, type=int, help='image height')
    args, _ = parser.parse_known_args()
    return args


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


def image_bgr_rgb(img):
    img_data = img[:, :, ::-1]
    return img_data


def img_process(img_, crop_size=513):
    """pre_process"""
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(IMAGE_MEAN)
    image_std = np.array(IMAGE_STD)
    img_ = (img_ - image_mean) / image_std
    img_ = image_bgr_rgb(img_)
    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
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


def main():
    args = _parse_args()

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'refinenet'
    infer_total_time = 0
    hist = np.zeros((args.num_classes, args.num_classes))
    with open(args.data_lst) as f:
        img_lst = f.readlines()

        os.makedirs(INFER_RESULT_DIR, exist_ok=True)
        for i, line in enumerate(img_lst):
            img_path, msk_path = line.strip().split(' ')
            img_path = os.path.join(args.data_root, img_path)
            msk_path = os.path.join(args.data_root, msk_path)
            print("The", i+1, "img_path:", img_path)

            # read image and mask
            img = cv2.imread(img_path)
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

            # preprocess
            ori_h, ori_w = img.shape[0], img.shape[1]
            img, resize_h, resize_w = img_process(img, 513)
            img = np.expand_dims(img, 0)  # NCHW
            img = np.ascontiguousarray(img)
            img = np.array(img).astype('float32')

            # infer
            if not send_source_data(0, img, stream_name, stream_manager_api):
                return
            # Obtain the inference result by specifying streamName and uniqueId.
            key_vec = StringVector()
            key_vec.push_back(b'modelInfer')
            start_time = time.time()
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            infer_total_time += time.time() - start_time
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            mask_image = res.reshape(1, args.num_classes, args.image_height, args.image_width)

            # flip
            img_f = img
            img_f = img_f[:, :, :, ::-1]
            if not send_source_data(0, img_f, stream_name, stream_manager_api):
                return
            # Obtain the inference result by specifying streamName and uniqueId.
            key_vec = StringVector()
            key_vec.push_back(b'modelInfer')
            start_time = time.time()
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            infer_total_time += time.time() - start_time
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            mask_image_f = res.reshape(1, args.num_classes, args.image_height, args.image_width)

            # post process
            image_result = np.squeeze(mask_image+mask_image_f[:, :, :, ::-1])
            probs_ = image_result[:, :resize_h, :resize_w].transpose((1, 2, 0))
            probs_ = cv2.resize(probs_, (ori_w, ori_h))
            result_mask = probs_.argmax(axis=2)

            # calculate accuracy
            hist += cal_hist(mask.flatten(), result_mask.flatten(), args.num_classes)

            # save result
            seg_name = INFER_RESULT_DIR + "/" + os.path.split(msk_path)[1]
            new_seg = mask_image
            new_seg = np.squeeze(new_seg)
            new_seg = new_seg.transpose(1, 2, 0)
            new_seg = new_seg.argmax(axis=2)
            new_seg = np.array(new_seg).astype('int32')
            new_seg[new_seg > 0] = 255
            cv2.imwrite(seg_name, new_seg)

    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('infer total time:', infer_total_time)
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))

    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
