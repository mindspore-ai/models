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
sample script of CLUE infer using SDK run in docker
"""

import argparse
import glob
import os
import time
from pathlib import Path

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

TP = 0
FP = 0
FN = 0
TN = 0
NegNum = 0
PosNum = 0


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="textrcnn process")
    parser.add_argument("--pipeline", type=str, default="textrcnn.pipeline", help="SDK infer pipeline")
    parser.add_argument('--dataset', type=str, default="MR", choices=['MR', 'SUBJ', 'SST2'])
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Dataset contain input_ids, label_ids")
    parser.add_argument("--label_file", type=str, default="../data/config/infer_label.txt", help="label ids to name")
    parser.add_argument("--output_file", type=str, default="output.txt", help="save result to file")
    parser.add_argument("--f1_method", type=str, default="BF1", help="calc F1 use the number label,(BF1, MF1)")
    parser.add_argument("--do_eval", type=bool, default=True, help="eval the accuracy of model ")
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
    # add tensor segment to line to tensor1
    tensor1 = tensor.reshape(-1, 50)
    tensor = tensor1[0]
    tensor = np.expand_dims(tensor, 0)
    # end add function
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
    send one stream to infer model, includes input ids.

    Returns:
        bool: send data success or not
    """
    input_ids = os.path.realpath(os.path.join(args.data_dir, args.dataset, '00_feature', file_name))
    if not send_source_data(0, input_ids, stream_name, stream_manager):
        return False
    return True


def read_label_file(label_file):
    """
    Args:
        label file
    Returns:
        label list
    """
    label_list = [line.strip() for line in open(label_file).readlines()]
    return label_list


def count_pred_result(args, file_name, logit_id, class_num=2, max_seq_length=128):
    """
    support two method to calc f1 sore, if dataset has two class, suggest using BF1,
    else more than two class, suggest using MF1.
    Args:
        args: param of config.
        file_name: label file name.
        logit_id: output tensor of infer.
        class_num: cluner data default is 2.
        max_seq_length: sentence input length default is 128.

    global:
        TP: pred == target == 1
        FP: pred == 1 target == 0
        FN: pred == 0 target == 1
        TN: pred == target == 0
    """
    file_name_index = file_name.split('_')[-1][:-4]
    label_file = os.path.realpath(os.path.join(args.data_dir, args.dataset, 'label_ids.npy'))
    real_label_index = int(file_name_index)
    label_ids = np.load(label_file)[real_label_index]
    print("real label is: ", label_ids)
    # label_ids.reshape(max_seq_length, -1)
    global TP, FP, FN, TN, NegNum, PosNum
    if args.f1_method == "BF1":

        pos_eva = np.isin(logit_id, [1]) # prediction
        pos_label = np.isin(label_ids, [1]) # target

        TP += np.sum(pos_eva & pos_label) # 1 1
        FP += np.sum(pos_eva & (~pos_label)) # 1 0
        FN += np.sum((~pos_eva) & pos_label) # 0 1
        TN += np.sum((~pos_eva) & (~pos_label)) # 0 0
        NegNum += np.sum(~pos_label)
        PosNum += np.sum(pos_label)
        print('TP= %d,    FP= %d,    FN= %d,    TN= %d' % (TP, FP, FN, TN))

    else:
        target = np.zeros((len(label_ids), class_num), dtype=np.int32)
        logit_id_str = map(str, logit_id)
        pred = np.zeros((len(logit_id_str), class_num), dtype=np.int32)
        for i, label in enumerate(label_ids):
            if label > 0:
                target[i][label] = 1
        for i, label in enumerate(logit_id):
            if label > 0:
                pred[i][label] = 1
        target = target.reshape(class_num, -1)
        pred = pred.reshape(class_num, -1)
        for i in range(0, class_num):
            for j in range(0, max_seq_length):
                if pred[i][j] == 1:
                    if target[i][j] == 1:
                        TP += 1
                    else:
                        FP += 1
                if pred[i][j] == 0:
                    if target[i][j] == 0:
                        TN += 1
                    else:
                        FN += 1


def post_process(args, file_name, infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 50.
    """
    # print the infer result
    # print("==============================================================")
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f2')
    # res = res.reshape(max_seq_length, -1)
    file_sn = file_name.split('.')[0]
    print(res)
    logit_id = np.argmax(res, axis=-1) # 结果
    label_list = read_label_file(os.path.realpath(args.label_file))
    infer_result = label_list[logit_id]
    print("ids : %s , infer output is: %s" % (file_sn, infer_result))
    filename_save = file_sn + '.txt'
    output_path = os.path.realpath(os.path.join(args.data_dir, args.dataset, 'output', filename_save))
    with open(output_path, "w") as file_:
        file_.write("{}\n".format(str(infer_result)))

    if args.do_eval:
        count_pred_result(args, file_name, logit_id)


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
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'im_textrcnnbase'
    infer_total_time = 0
    # input_ids file list, every file content a tensor[1,128]
    # os.path.realpath(os.path.join(args.data_dir,args.dataset,'ids',file_name))
    out_path = Path(os.path.join(os.path.realpath(args.data_dir), args.dataset, 'output'))
    out_path.mkdir(parents=True, exist_ok=True)
    file_list = glob.glob(os.path.join(os.path.realpath(args.data_dir), args.dataset, '00_feature', '*.bin'))
    file_list.sort()
    print(len(file_list))

    labels_path = os.path.join(os.path.realpath(args.data_dir), args.dataset, "labels")
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    label_file = os.path.realpath(os.path.join(args.data_dir, args.dataset, 'label_ids.npy'))

    for file_path in file_list:
        file_name = file_path.split("/")[-1]
        file_name_index = file_name.split('_')[-1][:-4]
        label_ids = np.load(label_file)[int(file_name_index)]

        label_path = os.path.join(labels_path, file_name)
        label_ids.tofile(label_path)

    for input_ids in file_list:
        file_name = input_ids.split('/')[-1]
        if not send_appsrc_data(args, file_name, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec) # 结果
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        post_process(args, file_name, infer_result)

    if args.do_eval:
        print("==============================================================")
        print('Totol ids input is: %d,    NegNum: %d,    PosNum: %d' % (len(file_list), NegNum, PosNum))
        print('TP=%d,    FP=%d,    FN= %d,    TN= %d' % (TP, FP, FN, TN))
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        print("Accuracy:  {:.6f} ".format(accuracy))
        precision = TP / (TP + FP)
        print("Precision:  {:.6f} ".format(precision))
        recall = TP / (TP + FN)
        print("Recall:  {:.6f} ".format(recall))
        print("F1:  {:.6f} ".format(2 * precision * recall / (precision + recall)))
        print("==============================================================")
    print("Infer images sum: {}, cost total time: {:.6f} sec.".format(len(file_list), infer_total_time))
    print("The throughput: {:.6f} bin/sec.".format(len(file_list) / infer_total_time))
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
