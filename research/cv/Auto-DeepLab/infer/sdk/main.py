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
# ===========================================================================
"""main.py"""
import argparse
import base64
import json
import os

import numpy as np

from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi

import utils


def sdk_args():
    """sdk_args"""
    parser = argparse.ArgumentParser("Auto-DeepLab SDK Inference")
    parser.add_argument('--pipeline', type=str, default='', help='path to pipeline file')
    # val data
    parser.add_argument('--data_root', type=str, default='',
                        help='root path of val data')
    parser.add_argument('--result_path', type=str, default='',
                        help='output_path')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='number of classes')
    args = parser.parse_args()
    return args


def _init_stream(pipeline_path):
    """_init_stream"""
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


def _do_infer(stream_manager_api, data_input):
    """_do_infer"""
    stream_name = b'segmentation'
    unique_id = stream_manager_api.SendDataWithUniqueId(
        stream_name, 0, data_input)
    if unique_id < 0:
        raise RuntimeError("Failed to send data to stream.")

    timeout = 7000
    infer_result = stream_manager_api.GetResultWithUniqueId(
        stream_name, unique_id, timeout)
    if infer_result.errorCode != 0:
        raise RuntimeError(
            "GetResultWithUniqueId error, errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))

    load_dict = json.loads(infer_result.data.decode())
    image_mask = load_dict["MxpiImageMask"][0]
    data_str = base64.b64decode(image_mask['dataStr'])
    shape = image_mask['shape']
    return np.frombuffer(data_str, dtype=np.uint8).reshape(shape)

def main():
    """main"""
    args = sdk_args()

    # init stream manager
    stream_manager_api = _init_stream(args.pipeline)
    if not stream_manager_api:
        exit(1)

    os.makedirs(args.result_path, exist_ok=True)
    data_input = MxDataInput()
    dataset = utils.CityscapesDataLoader(args.data_root)
    hist = np.zeros((args.num_classes, args.num_classes))
    for data_item in dataset:
        print(f"start infer {data_item['file_name']}")
        data_input.data = data_item['img']
        gtFine = utils.encode_segmap(data_item['gt'], 255)
        pred = _do_infer(stream_manager_api, data_input)

        hist += utils.fast_hist(pred.copy().flatten(), gtFine.flatten(), args.num_classes)
        color_mask_res = utils.label_to_color_image(pred)

        folder_path = os.path.join(args.result_path, data_item['file_path'].split(os.sep)[-1])
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        result_file = os.path.join(folder_path, data_item['file_name'].replace('leftImg8bit', 'pred_color'))
        color_mask_res.save(result_file)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    print("per-class IOU", iou)
    print("mean IOU", round(np.nanmean(iou) * 100, 2))

    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    main()
