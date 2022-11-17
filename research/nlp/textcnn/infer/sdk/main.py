"""
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

import argparse
import glob
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

eval_var = {"TP": 0, "FP": 0, "FN": 0, "NegNum": 0, "PosNum": 0}

sdkFile_path = os.path.realpath(os.path.dirname(__file__))


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="textcnn process")
    parser.add_argument("--pipeline",
                        type=str,
                        default="../data/config/textcnn.pipeline",
                        help="SDK infer pipeline")
    parser.add_argument("--data_dir",
                        type=str,
                        default="../data",
                        help="Dataset contain ids and labels")
    parser.add_argument("--label_file",
                        type=str,
                        default="../data/config/infer_label.txt",
                        help="label ids to name")
    parser.add_argument("--output_file",
                        type=str,
                        default="output.txt",
                        help="save result to file")
    parser.add_argument("--do_eval",
                        type=bool,
                        default=True,
                        help="eval the accuracy of model ")
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=51,
                        help="sentence length, default is 51.")
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(appsrc_id, filename, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor = np.fromfile(filename, dtype=np.int32)
    tensor = np.expand_dims(tensor, 0)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
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
    return True


def send_appsrc_data(args, file_name, stream_name, stream_manager):
    """
    send one stream (input ids ) to infer model.
    Returns:
        bool: send data success or not
    """
    input_ids = os.path.realpath(os.path.join(args.data_dir, 'ids', file_name))
    if not send_source_data(0, input_ids, stream_name, stream_manager):
        return False
    return True


def read_label_file(label_file):
    """
    Args:
        label_file:
        "0"
        "1"
        ...
    Returns:
        label list
    """
    label_list = [line.strip() for line in open(label_file).readlines()]
    return label_list


def count_pred_result(args, eval_var_param, file_name, logit_id):
    """
    Args:
        args: param of config.
        file_name: label file name.
        logit_id: output tensor of infer.

    global:
        TP: pred == target
        FP: in pred but not in target
        FN: in target but not in pred
    """
    label_file = os.path.realpath(
        os.path.join(sdkFile_path, args.data_dir, 'labels', file_name))
    label_ids = np.fromfile(label_file, np.int32)
    pos_eva = np.isin(logit_id, [1])
    pos_label = np.isin(label_ids, [1])
    eval_var_param["TP"] += np.sum(pos_eva & pos_label)
    eval_var_param["FP"] += np.sum(pos_eva & (~pos_label))
    eval_var_param["FN"] += np.sum((~pos_eva) & pos_label)
    eval_var_param["NegNum"] += np.sum(~pos_label)
    eval_var_param["PosNum"] += np.sum(pos_label)
    if (~pos_eva) & pos_label:
        print('*******************ids: %s is FN error' % (file_name))
    if pos_eva & (~pos_label):
        print('*******************ids: %s is FP error' % (file_name))


def post_process(args, eval_var_post, file_name, infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 128.
    """
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                        dtype='<f4')
    file_sn = file_name.split('.')[0]
    logit_id = np.argmax(res, axis=-1)
    label_list = read_label_file(os.path.join(sdkFile_path, args.label_file))
    infer_result = label_list[logit_id]
    print("Ids : %s , infer output is: %s" % (file_sn, infer_result))
    filename_save = file_sn + '.txt'
    output_path = os.path.realpath(
        os.path.join(sdkFile_path, '../output/sdk_result', filename_save))
    with open(output_path, "w") as result_file:
        result_file.write("{}\n".format(str(infer_result)))

    if args.do_eval:
        count_pred_result(args, eval_var_post, file_name, logit_id)


def run():
    """
    read pipeline and do infer
    """
    args = parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.join(sdkFile_path, args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'im_bertbase'
    infer_total_time = 0
    sdk_out_path = '../output/sdk_result'
    if not os.path.exists(sdk_out_path):
        os.makedirs(os.path.join(sdkFile_path, sdk_out_path))
    file_list = glob.glob(
        os.path.join(sdkFile_path, args.data_dir, 'ids', '*.bin'))
    file_list.sort()
    for input_ids in file_list:
        file_name = input_ids.split('/')[-1]
        if not send_appsrc_data(args, file_name, stream_name,
                                stream_manager_api):
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
            print("GetProtobuf error. errorCode=%d" %
                  (infer_result[0].errorCode))
            return
        post_process(args, eval_var, file_name, infer_result)

    if args.do_eval:
        print("==============================================================")
        print('Totol ids input is: %d,    NegNum: %d,    PosNum: %d' %
              (len(file_list), eval_var["NegNum"], eval_var["PosNum"]))
        print('TP=%d,    FP=%d,    FN= %d' %
              (eval_var["TP"], eval_var["FP"], eval_var["FN"]))
        accuracy = 1 - (eval_var["FP"] + eval_var["FN"]) / (
            eval_var["NegNum"] + eval_var["PosNum"])
        print("Accuracy:  {:.6f} ".format(accuracy))
        precision = eval_var["TP"] / (eval_var["TP"] + eval_var["FP"])
        print("Precision:  {:.6f} ".format(precision))
        recall = eval_var["TP"] / (eval_var["TP"] + eval_var["FN"])
        print("Recall:  {:.6f} ".format(recall))
        print("F1:  {:.6f} ".format(2 * precision * recall /
                                    (precision + recall)))
        print("==============================================================")
    print("Infer images sum: {}, cost total time: {:.6f} sec.".format(
        len(file_list), infer_total_time))
    print("The throughput: {:.6f} bin/sec.".format(
        len(file_list) / infer_total_time))
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
