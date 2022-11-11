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

"""
sample script of ternarybert infer using SDK run in docker
"""

import argparse
import glob
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

parser = argparse.ArgumentParser(description="ternaryBert inference")
parser.add_argument("--pipeline_file", type=str, required=True, help="SDK infer pipeline")
parser.add_argument("--data_dir", type=str, required=True, help="input data directory")
parser.add_argument("--res_dir", type=str, required=True, help="results directory")
parser.add_argument('--batch_size', type=int, default=32, help='batch size for infering')
parser.add_argument('--seq_length', type=int, default=128, help='sequence length')
args = parser.parse_args()

def send_source_data(appsrc_id, file_name, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    # transform file to tensor
    tensors = np.fromfile(file_name, dtype=np.int32).reshape([args.batch_size, args.seq_length])
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()

    for i in range(args.batch_size):
        tensor = np.expand_dims(tensors[i, :], 0)
        tensor_package = tensor_package_list.tensorPackageVec.add()
        tensor_vec = tensor_package.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(tensor.shape)
        tensor_vec.tensorDataType = 3 # int32
        array_bytes = tensor.tobytes()
        tensor_vec.dataStr = array_bytes
        tensor_vec.tensorDataSize = tensor.shape[0]

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
    send three stream to infer model, include input_ids, token_type_id and input_mask.

    Returns:
        bool: send data success or not
    """
    input_ids_path = os.path.realpath(os.path.join(args.data_dir, "00_input_ids", file_name))
    if not send_source_data(0, input_ids_path, stream_name, stream_manager):
        return False

    token_type_id_path = os.path.realpath(os.path.join(args.data_dir, "01_token_type_id", file_name))
    if not send_source_data(1, token_type_id_path, stream_name, stream_manager):
        return False

    input_mask_path = os.path.realpath(os.path.join(args.data_dir, "02_input_mask", file_name))
    if not send_source_data(2, input_mask_path, stream_name, stream_manager):
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
    with open(res_file, 'ab') as f:
        for k in range(args.batch_size):
            f.write(result.tensorPackageVec[k].tensorVec[13].dataStr)


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

    stream_name = b'im_ternarybert'
    infer_total_time = 0
    # get all files endwith 'bin'
    file_list = glob.glob(os.path.join(os.path.realpath(os.path.join(args.data_dir, "00_input_ids")), "*.bin"))
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

    print("Infer images sum: {}, cost total time: {:.6f} sec.".format(len(file_list), infer_total_time))
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
