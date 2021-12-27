# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import json
import datetime
from StreamManagerApi import StreamManagerApi, MxDataInput


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="east process")
    parser.add_argument("--pipeline", type=str, default='./pipeline/east.pipeline', help="SDK infer pipeline")
    parser.add_argument("--image_path", type=str, default='../data/image/', help="root path of image")
    parser.add_argument('--result_path', default='./result', type=str,
                        help='the folder to save the semantic mask images')
    args_opt = parser.parse_args()
    return args_opt


def run():
    args = parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(args.pipeline, 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    # Construct the input of the stream
    data_input = MxDataInput()

    file_list = os.listdir(args.image_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    start_time = datetime.datetime.now()
    for file_name in file_list:
        start_time = datetime.datetime.now()
        print(file_name)
        file_path = os.path.join(args.image_path, file_name)
        if file_name.endswith(".JPG") or file_name.endswith(".jpg"):
            with open(file_path, 'rb') as f:
                data_input.data = f.read()

        # Inputs data to a specified stream based on streamName.
        stream_name = b'classification+detection'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendDataWithUniqueId(
            stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            return

        # Obtain the inference result by specifying streamName and uniqueId
        infer_result = stream_manager_api.GetResultWithUniqueId(stream_name, unique_id, 3000)
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            return

        load_dict = json.loads(infer_result.data.decode())
        result = []
        if load_dict.get('MxpiTextObject') is None:
            result = []
        else:
            result = load_dict['MxpiTextObject']

        boxes = []
        for res in result:
            print(res)
            boxes.append([int(res['x0']), int(res['y0']), int(res['x1']), int(res['y1']),
                          int(res['x2']), int(res['y2']), int(res['x3']), int(res['y3'])])

        if boxes is not None:
            seq = []
            seq.extend([','.join([str(int(b))
                                  for b in box]) + '\n' for box in boxes])
        with open(os.path.join(args.result_path, 'res_' +
                               os.path.basename(file_name).replace('.jpg', '.txt')), 'w') as f:
            f.writelines(seq)

    end_time = datetime.datetime.now()
    print('EAST sdk run time: {}'.format(end_time - start_time))

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
