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
""" main.py """
import os
import argparse
import datetime
import json
import numpy as np
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType


shape = [1, 3, 32, 32]


def parse_args(parsers):
    """
    Parse commandline arguments.
    """
    parsers.add_argument('--images_txt_path', type=str, default="../data/preprocess_Result/label.txt",
                         help='image path')
    return parsers


def read_file_list(input_f):
    """
    :param infer file content:
        1.bin 0
        2.bin 2
        ...
    :return image path list, label list
    """
    image_file_l = []
    label_l = []
    if not os.path.exists(input_f):
        print('input file does not exists.')
    with open(input_f, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(',')
            files = line[0]
            label = int(line[1])
            image_file_l.append(files)
            label_l.append(label)
    return image_file_l, label_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Om PDarts Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/pdarts.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    res_dir_name = 'result'
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists("../results"):
        os.makedirs("../results")

    file_list, label_list = read_file_list(args.images_txt_path)

    img_size = len(file_list)
    results = []

    for idx, file in enumerate(file_list):
        image_path = os.path.join(args.images_txt_path.replace('label.txt', '00_img_data'), file)

        # Construct the input of the stream
        data_input = MxDataInput()
        with open(image_path, 'rb') as f:
            data = f.read()
        data_input.data = data
        tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
        tensorVec1 = tensorPackage1.tensorVec.add()
        tensorVec1.deviceId = 0
        tensorVec1.memType = 0
        for t in shape:
            tensorVec1.tensorShape.append(t)
        tensorVec1.dataStr = data_input.data
        tensorVec1.tensorDataSize = len(data)
        protobufVec1 = InProtobufVector()
        protobuf1 = MxProtobufIn()
        protobuf1.key = b'appsrc0'
        protobuf1.type = b'MxTools.MxpiTensorPackageList'
        protobuf1.protobuf = tensorPackageList1.SerializeToString()
        protobufVec1.push_back(protobuf1)

        unique_id = stream_manager.SendProtobuf(b'pdarts', b'appsrc0', protobufVec1)

        # Obtain the inference result by specifying streamName and uniqueId.
        start_time = datetime.datetime.now()
        infer_result = stream_manager.GetResult(b'pdarts', 0)
        # Obtain the inference result by specifying streamName and uniqueId.
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        infer_res = infer_result.data.decode()
        print("process img: {}, infer result: {}".format(file, infer_res))
        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiClass') is None:
            with open(res_dir_name + "/" + file.split(".")[0] + '.txt', 'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict.get('MxpiClass')
        res = [i['classId'] for i in load_dict['MxpiClass']]
        results.append(res)

        with open(res_dir_name + "/" + file.split(".")[0] + '_1.txt', 'w') as f_write:
            res_list = [str(item.get("classId")) + " " for item in res_vec]
            f_write.writelines(res_list)
            f_write.write('\n')

    results = np.array(results)
    labels = np.array(label_list)
    np.savetxt("./results/infer_results.txt", results, fmt='%s')
    # destroy streams
    stream_manager.DestroyAllStreams()
