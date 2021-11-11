# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import os
import sys
import math

import numpy as np
import cv2

from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput
from StreamManagerApi import StringVector
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import InProtobufVector
import MxpiDataType_pb2 as MxpiDataType

H, W = 126, 126


def decode_image(img_o):
    mean = 0.5 * 255
    std = 0.5 * 255
    return (img_o * std + mean).astype(np.uint8).transpose(
        (1, 2, 0))


def calc_psnr(img1: np.ndarray, img2: np.ndarray, crop_border: int = 0, test_on_y=False):
    """
    Calc psnr between img1 and img2.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_on_y:
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img1 = np.expand_dims(img1, axis=2)
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2 = np.expand_dims(img2, axis=2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def calc_ssim(img1: np.ndarray, img2: np.ndarray, crop_border):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border >= 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssimes = []
    for index in range(img1.shape[2]):
        ssimes.append(_ssim(img1[..., index], img2[..., index]))
    return np.array(ssimes).mean()


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/srgan_opencv.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = sys.argv[1]
    out_name = sys.argv[2]
    gt_name = sys.argv[3]

    file_list = os.listdir(dir_name)

    if not os.path.exists(out_name):
        os.makedirs(out_name)

    psnrs, ssims, psnrs_y = [], [], []
    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        save_path = os.path.join(out_name, file_name)
        gt_path = os.path.join(gt_name, file_name)
        if not (file_name.lower().endswith(".jpg")
                or file_name.lower().endswith(".jpeg")
                or file_name.lower().endswith(".png")):
            continue

        empty_data = []
        stream_name = b'srgan_opencv'
        in_plugin_id = 0
        input_key = 'appsrc0'

        img = cv2.imread(file_path)

        h, w, c = img.shape

        hs = math.ceil(H/h)
        ws = math.ceil(W/w)

        img_pad = np.zeros((hs*h, ws*w, 3), np.uint8)

        for i in range(ws):
            for j in range(hs):
                if i % 2 == 1:
                    img_f = cv2.flip(img, 1)
                else:
                    img_f = img
                if j % 2 == 1:
                    img_f = cv2.flip(img_f, 0)
                else:
                    img_f = img_f
                img_pad[h*j: h*(j+1), w*i: w*(i+1)] = img_f

        img_pad = img_pad[:H, :W]

        img = img_pad[:, :, ::-1].transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.array((img-127.5)/127.5).astype(np.float32)

        tensor_list = MxpiDataType.MxpiTensorPackageList()
        tensor_pkg = tensor_list.tensorPackageVec.add()

        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(img.shape)
        tensor_vec.tensorDataType = 0
        tensor_vec.dataStr = img.tobytes()
        tensor_vec.tensorDataSize = len(img)
        buf_type = b"MxTools.MxpiTensorPackageList"

        protobuf = MxProtobufIn()
        protobuf.key = input_key.encode("utf-8")
        protobuf.type = buf_type
        protobuf.protobuf = tensor_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)
        err_code = stream_manager_api.SendProtobuf(
            stream_name, in_plugin_id, protobuf_vec)
        if err_code != 0:
            print(
                "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
                "buf_type(%s), err_code(%s).", stream_name, in_plugin_id,
                input_key, buf_type, err_code)

        start_time = datetime.datetime.now()
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        TensorList = MxpiDataType.MxpiTensorPackageList()
        TensorList.ParseFromString(infer_result[0].messageBuf)
        data = np.frombuffer(
            TensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        data = data.reshape(3, 504, 504)
        img = decode_image(data)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img[:h*4, :w*4]

        cv2.imwrite(save_path, img)

        gt_img = cv2.imread(gt_path)

        psnr = calc_psnr(img, gt_img, 4)
        psnr_y = calc_psnr(img, gt_img, 4, True)

        ssim = calc_ssim(img, gt_img, 4)

        psnrs.append(psnr)
        ssims.append(ssim)
        psnrs_y.append(psnr_y)

        print(f"[{file_name}]psnr:{psnr}-ssim:{ssim}-psnr_y:{psnr_y}")

    print(
        f'avg_psnr:{sum(psnrs)/len(psnrs)},avg_ssim:{sum(ssims)/len(ssims)},avg_psnr_y:{sum(psnrs_y)/len(psnrs_y)}')

    # destroy streams
    stream_manager_api.DestroyAllStreams()
