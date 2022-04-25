# Copyright (c) 2022. Huawei Technologies Co., Ltd
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
sample script of MMoE infer using SDK run in docker
"""

import argparse
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
            MxProtobufIn, StringVector

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description='MMoE process')
    parser.add_argument('--data_dir', type=str, default='../data/input', help='Data path')
    parser.add_argument('--data_file', type=str, default='data_{}.npy')
    parser.add_argument('--income_file', type=str, default='income_labels_{}.npy')
    parser.add_argument('--married_file', type=str, default='married_labels_{}.npy')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--num_features', type=int, default=499, help='dim of feature')
    parser.add_argument('--num_labels', type=int, default=2, help='dim of label')
    parser.add_argument('--output_dir', type=str, default='./output', help='Data path')
    parser.add_argument('--pipeline', type=str, default='../data/config/MMoE.pipeline', help='SDK infer pipeline')
    args_opt = parser.parse_args()
    return args_opt

args = parse_args()

def send_source_data(appsrc_id, file_name, file_data, stream_name, stream_manager, shape, tp):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
    bool: send data success or not
    """
    tensors = np.array(file_data, dtype=tp).reshape(shape)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    data_input = MxDataInput()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensors.shape:
        tensor_vec.tensorShape.append(i)
    array_bytes = tensors.tobytes()
    data_input.data = array_bytes
    tensor_vec.dataStr = data_input.data
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
    print("Send successfully!")
    return True

def send_appsrc_data(appsrc_id, file_name, file_data, stream_name, stream_manager, shape, tp):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
    bool: send data success or not
    """
    if not send_source_data(appsrc_id, file_name, file_data, stream_name, stream_manager, shape, tp):
        return False
    return True

def post_process(infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        infer_result: get logit from infer result
    """
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    income_pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16)
    income_pred = income_pred.reshape((-1, 2))
    married_pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float16)
    married_pred = married_pred.reshape((-1, 2))
    return income_pred, married_pred

def get_auc(labels, preds):
    labels = labels.flatten().tolist()
    preds = preds.flatten().tolist()
    return roc_auc_score(labels, preds)

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
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # prepare data
    data = np.load(os.path.join(args.data_dir, args.data_file.format(args.mode))).astype(np.float16)
    income = np.load(os.path.join(args.data_dir, args.income_file.format(args.mode))).astype(np.float16)
    married = np.load(os.path.join(args.data_dir, args.married_file.format(args.mode))).astype(np.float16)

    if(data.shape[0] != income.shape[0] or income.shape[0] != married.shape[0]):
        print("number of input data not completely equal")
        exit()
    rows = data.shape[0]

    # statistical variable
    income_labels = []
    married_labels = []
    income_preds = []
    married_preds = []
    infer_total_time = 0

    # write predict results
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for i in range(rows):
        # fetch data
        data_batch = data[i]
        income_batch = income[i]
        married_batch = married[i]

        # data shape
        data_shape = (-1, args.num_features)

        # data type
        data_type = np.float16

        # send data
        stream_name = b'MMoE'
        if not send_appsrc_data(0, 'data', data_batch, stream_name, stream_manager_api, data_shape, data_type):
            return

        # Obtain the inference result by specifying streamName and uniqueId.
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

        # updata variable
        income_pred, married_pred = post_process(infer_result)
        income_preds.extend(income_pred)
        married_preds.extend(married_pred)
        income_labels.extend(income_batch)
        married_labels.extend(married_batch)

    income_preds = np.array(income_preds)
    married_preds = np.array(married_preds)
    income_labels = np.array(income_labels)
    married_labels = np.array(married_labels)
    np.save(os.path.join(args.output_dir, 'income_preds_{}.npy'.format(args.mode)), income_preds)
    np.save(os.path.join(args.output_dir, 'married_preds_{}.npy'.format(args.mode)), married_preds)
    np.save(os.path.join(args.output_dir, 'income_labels_{}.npy').format(args.mode), income_labels)
    np.save(os.path.join(args.output_dir, 'married_labels_{}.npy'.format(args.mode)), married_labels)
    income_auc = get_auc(income_labels, income_preds)
    married_auc = get_auc(married_labels, married_preds)
    print('<<========  Infer Metric ========>>')
    print('Mode: {}'.format(args.mode))
    print('Number of samples: {}'.format(rows))
    print('Total inference time: {}'.format(infer_total_time))
    print('Average inference time: {}'.format(infer_total_time/rows))
    print('Income auc: {}'.format(income_auc))
    print('Married auc: {}'.format(married_auc))
    print('<<===============================>>')
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    run()
