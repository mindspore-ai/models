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

"""
sample script of atae-lstm infer using SDK run in docker
"""

import argparse
import glob
import os
import time
import ast

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

parser = argparse.ArgumentParser(description="atae-lstm inference")
parser.add_argument("--pipeline_file", type=str, required=True, help="SDK infer pipeline")
parser.add_argument("--data_dir", type=str, required=True, help="input data directory")
parser.add_argument("--res_dir", type=str, required=True, help="results directory")
parser.add_argument("--do_eval", type=ast.literal_eval, default=True, help="eval the accuracy of model")
args = parser.parse_args()

def send_source_data(appsrc_id, file_name, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    # transform file to tensor
    tensor_batch = 1
    tensor_len = 50
    if appsrc_id == 0:
        tensor = np.fromfile(file_name, dtype=np.int32).reshape([tensor_batch, tensor_len])
    else:
        tensor = np.fromfile(file_name, dtype=np.int32).reshape([tensor_batch])

    # create MxpiTensorPackageList
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


def send_appsrc_data(file_name, stream_name, stream_manager):
    """
    send three stream to infer model, include content, sen_len and aspect.

    Returns:
        bool: send data success or not
    """
    content_path = os.path.realpath(os.path.join(args.data_dir, "00_content", file_name))
    if not send_source_data(0, content_path, stream_name, stream_manager):
        return False

    sen_len_path = os.path.realpath(os.path.join(args.data_dir, "01_sen_len", file_name))
    if not send_source_data(1, sen_len_path, stream_name, stream_manager):
        return False

    aspect_path = os.path.realpath(os.path.join(args.data_dir, "02_aspect", file_name))
    if not send_source_data(2, aspect_path, stream_name, stream_manager):
        return False
    return True


def save_result(file_name, infer_result):
    """
    save the result of infer tensor.
    Args:
        file_name: label file name.
        infer_result: get logit from infer result
    """
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)

    res_file = os.path.realpath(os.path.join(args.res_dir, file_name))
    with open(res_file, 'wb') as f:
        f.write(result.tensorPackageVec[0].tensorVec[0].dataStr)

    polarity = ['negative', 'neutral', 'positive']
    result_numpy = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, np.float32).reshape([1, 3])
    polarity_result = np.argmax(result_numpy)
    predict_file = os.path.realpath(os.path.join(args.res_dir, 'predict.txt'))
    with open(predict_file, 'a+') as f:
        f.write(file_name + ' infer result is: ' + polarity[polarity_result] + '\n')


def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline_file), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    stream_name = b'im_atae_lstm'
    infer_total_time = 0
    # get all files endwith 'bin'
    file_list = glob.glob(os.path.join(os.path.realpath(os.path.join(args.data_dir, "00_content")), "*.bin"))
    for input_ids in file_list:
        # send appsrc data
        file_name = input_ids.split('/')[-1]
        if not send_appsrc_data(file_name, stream_name, stream_manager_api):
            return
        # obtain the inference result
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        save_result(file_name, infer_result)

    if args.do_eval:
        # compute acc
        file_names = []
        for files in os.walk(args.data_dir):
            file_names = files[2]
        file_num = len(file_names)
        correct = 0
        for f in file_names:
            label_path = os.path.join(args.data_dir, "solution_path", f)
            result_path = os.path.join(args.res_dir, f)

            label_numpy = np.fromfile(label_path, np.float32).reshape([1, 3])
            polarity_label = np.argmax(label_numpy)
            result_numpy = np.fromfile(result_path, np.float32).reshape([1, 3])
            polarity_result = np.argmax(result_numpy)
            if polarity_result == polarity_label:
                correct += 1
        acc = correct / float(file_num)
        print("\n---accuracy:", acc, "---\n")

    print("Infer images sum: {}, cost total time: {:.6f} sec.".format(len(file_list), infer_total_time))
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
