# !/usr/bin/env python

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
Sdk internece
"""
import argparse
import os
import time
import ast
import numpy as np

from PIL import Image
from pycocotools.coco import COCO

from api.infer import SdkApi
from api.postprocess import DetectionEngine
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StringVector
from config import config as cfg


def parser_args():
    """
    configuration parameter, input from outside
    """
    parser = argparse.ArgumentParser(description="yolov5 inference")
    parser.add_argument("--pipeline_path", type=str, required=False, default="config/yolov5.pipeline",
                        help="pipeline file path. The default is 'config/centernet.pipeline'. ")

    parser.add_argument('--nms_thresh', type=float, default=0.6, help='threshold for NMS')
    parser.add_argument('--ann_file', type=str, default='', help='path to annotation')
    parser.add_argument('--ignore_threshold', type=float, default=0.001, help='threshold to throw low quality boxes')

    parser.add_argument('--dataset_path', type=str, default='', help='path of image dataset')
    parser.add_argument('--result_files', type=str, default='./result', help='path to 310 infer result path')
    parser.add_argument('--multi_label', type=ast.literal_eval, default=True, help='whether to use multi label')
    parser.add_argument('--multi_label_thresh', type=float, default=0.1, help='threshold to throw low quality boxes')

    arg = parser.parse_args()
    return arg


def process_img(img_file):
    """
    Preprocessing the images
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = Image.open(img_file).convert("RGB")
    img = img.resize((cfg.MODEL_HEIGHT, cfg.MODEL_WIDTH), 0)
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = np.concatenate((img[..., ::2, ::2], img[..., 1::2, ::2], img[..., ::2, 1::2], img[..., 1::2, 1::2]), axis=1)

    return img


def image_inference(pipeline_path, stream_name, img_dir, detection):
    """
    image inference: get inference for images
    """
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        return

    img_data_plugin_id = 0
    print(f"\nBegin to inference for {img_dir}.\n")

    file_list = os.listdir(img_dir)
    coco = COCO(args.ann_file)
    start_time = time.time()

    for file_name in file_list:
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue

        img_ids_name = file_name.split('.')[0]
        img_id_ = int(np.squeeze(img_ids_name))
        imgIds = coco.getImgIds(imgIds=[img_id_])
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        image_shape = ((img['width'], img['height']),)
        img_id_ = (np.squeeze(img_ids_name),)

        imgs = process_img(os.path.join(img_dir, file_name))
        sdk_api.send_tensor_input(stream_name,
                                  img_data_plugin_id, "appsrc0",
                                  imgs.tobytes(), imgs.shape, cfg.TENSOR_DTYPE_FLOAT32)

        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = sdk_api. get_protobuf(stream_name, 0, keyVec)

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        output_small = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                                     dtype='float32').reshape((1, 20, 20, 3, 85))
        output_me = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr,
                                  dtype='float32').reshape((1, 40, 40, 3, 85))
        output_big = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr,
                                   dtype='float32').reshape((1, 80, 80, 3, 85))
        print('process {}...'.format(file_name))
        detection.detect([output_small, output_me, output_big], 1, image_shape, img_id_)

    print('do_nms_for_results...')
    detection.do_nms_for_results()
    detection.write_result()

    cost_time = time.time() - start_time
    print('testing cost time {:.2f}h'.format(cost_time / 3600.))


if __name__ == "__main__":
    args = parser_args()
    detections = DetectionEngine(args)
    stream_name0 = cfg.STREAM_NAME.encode("utf-8")
    print("stream_name0:")
    print(stream_name0)
    image_inference(args.pipeline_path, stream_name0, args.dataset_path, detections)
