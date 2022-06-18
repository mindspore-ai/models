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
# ============================================================================

import argparse
import glob
import json
import os
from contextlib import ExitStack
import cv2
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
        MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def preprocess(in_file):

    input_size = (320, 320)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imgbgr = cv2.imread(in_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=input_size, fx=1, fy=1)  # transforms.Resize(320)
    img = np.array(img, dtype=np.float32)
    img = center_crop(img, 256, 256)   # transforms.CenterCrop(256)
    img = img / 255.  # transforms.ToTensor()

    img[..., 0] = (img[..., 0] - mean[0]) / std[0]
    img[..., 1] = (img[..., 1] - mean[1]) / std[1]
    img[..., 2] = (img[..., 2] - mean[2]) / std[2]

    img = img.transpose(2, 0, 1)   # HWC -> CHW
    return img



class GlobDataLoader():
    def __init__(self, glob_pattern, limit=None):
        self.glob_pattern = glob_pattern
        self.limit = limit
        self.file_list = self.get_file_list()
        self.cur_index = 0

    def get_file_list(self):
        return glob.iglob(self.glob_pattern)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index == self.limit:
            raise StopIteration()
        label = None
        file_path = next(self.file_list)
        with open(file_path, 'rb') as fd:
            data = fd.read()

        self.cur_index += 1
        return get_file_name(file_path), label, data


class Predictor():
    def __init__(self, pipeline_conf, stream_name):
        self.pipeline_conf = pipeline_conf
        self.stream_name = stream_name

    def __enter__(self):
        self.stream_manager_api = StreamManagerApi()
        ret = self.stream_manager_api.InitManager()
        if ret != 0:
            raise Exception(f"Failed to init Stream manager, ret={ret}")

        # create streams by pipeline config file
        with open(self.pipeline_conf, 'rb') as f:
            pipeline_str = f.read()
        ret = self.stream_manager_api.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise Exception(f"Failed to create Stream, ret={ret}")
        self.data_input = MxDataInput()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # destroy streams
        self.stream_manager_api.DestroyAllStreams()

    def predict(self, dataset):
        print("Start predict........")
        print('>' * 30)
        for name, _, data in dataset:
            self.data_input.data = data
            yield self._predict(name, self.data_input)
        print("predict end.")
        print('<' * 30)

    def _predict(self, name, data):
        protobuf_data = self._predict_gen_protobuf(name)
        self._predict_send_protobuf(self.stream_name, 0, protobuf_data)
        result = self._predict_get_result(self.stream_name, 0)
        return name, json.loads(result.data.decode())

    def _predict_gen_protobuf(self, name):
        args = parse_args()
        file_path = os.path.join(args.glob, name+'.JPEG')

        img_np = preprocess(file_path)
        print("*********file_path is: ", file_path)

        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 0
        vision_vec.visionInfo.width = 256
        vision_vec.visionInfo.height = 256
        vision_vec.visionInfo.widthAligned = 256
        vision_vec.visionInfo.heightAligned = 256

        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = img_np.tobytes()
        vision_vec.visionData.dataSize = len(img_np)

        protobuf = MxProtobufIn()
        protobuf.key = b"appsrc0"
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec = InProtobufVector()

        protobuf_vec.push_back(protobuf)
        return protobuf_vec

    def _predict_send_protobuf(self, stream_name, in_plugin_id, data):
        self.stream_manager_api.SendProtobuf(stream_name, in_plugin_id, data)

    def _predict_send_data(self, stream_name, in_plugin_id, data_input):
        unique_id = self.stream_manager_api.SendData(stream_name, in_plugin_id,
                                                     data_input)
        if unique_id < 0:
            raise Exception("Failed to send data to stream")
        return unique_id

    def _predict_get_result(self, stream_name, unique_id):
        result = self.stream_manager_api.GetResult(stream_name, unique_id)
        if result.errorCode != 0:
            raise Exception(
                f"GetResultWithUniqueId error."
                f"errorCode={result.errorCode}, msg={result.data.decode()}")
        return result


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path.rstrip('/')))[0]


def result_encode(file_name, result):
    sep = ','
    pred_class_ids = sep.join(
        str(i.get('classId')) for i in result.get("MxpiClass", []))
    return f"{file_name} {pred_class_ids}\n"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--glob', help='img pth glob pattern.')
    parser.add_argument('--result_file', help='result file')
    return parser.parse_args()


def main():
    pipeline_conf = "../data/config/ResNeSt50.pipeline"
    stream_name = b'ResNeSt50'

    args = parse_args()
    result_fname = get_file_name(args.result_file)
    pred_result_file = f"{result_fname}.txt"
    dataset = GlobDataLoader(args.glob+"/*", limit=50000)
    print("dataset = ", dataset)
    with ExitStack() as stack:
        predictor = stack.enter_context(Predictor(pipeline_conf, stream_name))
        result_fd = stack.enter_context(open(pred_result_file, 'w'))
        for fname, pred_result in predictor.predict(dataset):
            result_fd.write(result_encode(fname, pred_result))

    print(f"success, result in {pred_result_file}")


if __name__ == "__main__":
    main()
