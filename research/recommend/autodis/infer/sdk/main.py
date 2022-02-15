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
sample script of autodis infer using SDK run in docker
"""

import argparse
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description='autodis process')
    parser.add_argument('--data_dir', type=str, default='../data/input', help='Data path')
    parser.add_argument('--ids_file', type=str, default='ids')
    parser.add_argument('--wts_file', type=str, default='wts')
    parser.add_argument('--label_file', type=str, default='label')
    parser.add_argument('--input_format', type=str, default='bin')
    parser.add_argument('--output_dir', type=str, default='./output', help='Data path')
    parser.add_argument('--pipeline', type=str, default='../data/config/autodis.pipeline', help='SDK infer pipeline')
    parser.add_argument('--dense_dim', type=int, default=13)
    parser.add_argument('--slot_dim', type=int, default=26)
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

def get_acc(labels, preds):
    """Accuracy"""
    accuracy = np.sum(labels == preds) / len(labels)
    return accuracy

def post_process(infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        infer_result: get logit from infer result
    """
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)
    res = res.reshape((-1,))
    label = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32)
    label = res.reshape((-1,))
    pred_label = np.round(res)
    return int(label[0]), res[0], int(pred_label[0])

def get_auc(labels, preds, n_bins=10000):
    """ROC_AUC"""
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    if total_case == 0:
        return 0
    pos_histogram = [0 for _ in range(n_bins+1)]
    neg_histogram = [0 for _ in range(n_bins+1)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins+1):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]
    return satisfied_pair / float(total_case)

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

    # preprocess data
    if args.input_format == 'txt':
        ids_data = np.loadtxt(os.path.join(args.data_dir, args.ids_file+"."+args.input_format), delimiter="\t")
        wts_data = np.loadtxt(os.path.join(args.data_dir, args.wts_file+"."+args.input_format), delimiter="\t")
        label_data = np.loadtxt(os.path.join(args.data_dir, args.label_file+"."+args.input_format), delimiter="\t")
    else:
        ids_data = np.fromfile(os.path.join(args.data_dir, args.ids_file+"."+args.input_format), dtype=np.int32)
        ids_data.shape = -1, 39
        wts_data = np.fromfile(os.path.join(args.data_dir, args.wts_file+"."+args.input_format), dtype=np.float32)
        wts_data.shape = -1, 39
        label_data = np.fromfile(os.path.join(args.data_dir, args.label_file+"."+args.input_format), dtype=np.float32)
        label_data.shape = -1, 1

    if(ids_data.shape[0] != wts_data.shape[0] or wts_data.shape[0] != label_data.shape[0]):
        print("number of input data not completely equal")
        exit()
    rows = label_data.shape[0]

    # statistical variable
    labels = []
    probs = []
    preds = []
    infer_total_time = 0

    # write predict label
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    fo = open(os.path.join(args.output_dir, "result.txt"), "w")
    fo.write("label\tprob\tpred\n")
    for i in range(rows):
        # fetch data
        ids = ids_data[i]
        wts = wts_data[i]
        label = label_data[i]

        # data shape
        ids_shape = (-1, args.dense_dim+args.slot_dim)
        wts_shape = (-1, args.dense_dim+args.slot_dim)
        label_shape = (-1, 1)

        # data type
        ids_type = np.int32
        wts_type = np.float32
        label_type = np.float32

        # send data
        stream_name = b'autodis'
        if not send_appsrc_data(0, "ids", ids, stream_name, stream_manager_api, ids_shape, ids_type):
            return
        if not send_appsrc_data(1, "wts", wts, stream_name, stream_manager_api, wts_shape, wts_type):
            return
        if not send_appsrc_data(2, "label", label, stream_name, stream_manager_api, label_shape, label_type):
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
        label_, prob_, pred_ = post_process(infer_result)
        label_ = label
        labels.append(label_)
        probs.append(prob_)
        preds.append(pred_)

        # write predict label
        fo.write(str(label_)+"\t"+str(prob_)+"\t"+str(pred_)+"\n")

    labels = np.array(labels)
    probs = np.array(probs)
    preds = np.array(preds)
    infer_acc = get_acc(labels, preds)
    infer_auc = get_auc(labels, probs)
    fo1 = open(os.path.join(args.output_dir, "metric.txt"), "w")
    fo1.write("Number of samples:%d\n"%(rows))
    fo1.write("Infer total time:%f\n"%(infer_total_time))
    fo1.write("Average infer time:%f\n"%(infer_total_time/rows))
    fo1.write("Infer acc:%f\n"%(infer_acc))
    fo1.write("Infer auc:%f\n"%(infer_auc))
    fo.close()
    fo1.close()
    print('<<========  Infer Metric ========>>')
    print("Number of samples:%d"%(rows))
    print("Infer total time:%f"%(infer_total_time))
    print("Average infer time:%f\n"%(infer_total_time/rows))
    print("Infer acc:%f"%(infer_acc))
    print("infer auc:%f"%(infer_auc))
    print('<<===============================>>')
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    run()
