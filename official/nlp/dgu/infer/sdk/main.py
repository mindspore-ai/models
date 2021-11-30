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
sample script of dgu infer using SDK run in docker
"""

import argparse
import glob
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector


def softmax(z):
    """
    softmax function
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


class Accuracy():
    """
    calculate accuracy
    """
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
    def update(self, logits, labels):
        labels = np.reshape(labels, -1)
        self.acc_num += np.sum(labels == logits)
        self.total_num += len(labels)


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="dgu process")
    parser.add_argument("--pipeline", type=str, default="", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
    parser.add_argument("--label_file", type=str, default="", help="label ids to name")
    parser.add_argument("--output_file", type=str, default="", help="save result to file")
    parser.add_argument("--task_name", type=str, default="atis_intent", help="(atis_intent, mrda, swda)")
    parser.add_argument("--do_eval", type=str, default="true", help="eval the accuracy of model")
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


def send_appsrc_data(file_name, stream_name, stream_manager):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
        bool: send data success or not
    """
    input_ids = os.path.realpath(os.path.join(args.data_dir, "00_data", file_name))
    if not send_source_data(0, input_ids, stream_name, stream_manager):
        return False
    input_mask = os.path.realpath(os.path.join(args.data_dir, "01_data", file_name))
    if not send_source_data(1, input_mask, stream_name, stream_manager):
        return False
    token_type_id = os.path.realpath(os.path.join(args.data_dir, "02_data", file_name))
    if not send_source_data(2, token_type_id, stream_name, stream_manager):
        return False
    return True


def read_label_file(label_file):
    """
    Args:
        label_file:
        "aa 3"
    Returns:
        label dic
    """
    label_map = {}
    for line in open(label_file).readlines():
        label, index = line.strip('\n').split('\t')
        label_map[index] = label
    return label_map


def process_infer(logit_id):
    """
    find label and position from the logit_id tensor.

    Args:
        logit_id: shape is [num_labels], example: [0..0.1..0].
    Returns:
        type of label: Q
    """
    result_label = label_dic[str(logit_id[0])]
    return result_label


def post_process(file_name, infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        file_name: label file name.
        infer_result: get logit from infer result
    """
    # print the infer result
    print("==============================================================")
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    logit_id = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    print("output tensor is: ", logit_id.shape)
    print("post_process:")
    logit_id = np.argmax(logit_id, axis=-1)
    logit_id = np.reshape(logit_id, -1)

    #output to file
    result_label = process_infer(logit_id)
    print(result_label)
    with open(args.output_file, "a") as file:
        file.write("{}: {}\n".format(file_name, str(result_label)))
    return logit_id


def run():
    """
    read pipeline and do infer
    """
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

    stream_name = b'im_dgu'
    infer_total_time = 0
    # input_ids file list
    file_list = glob.glob(os.path.join(os.path.realpath(args.data_dir), "00_data", "*.bin"))
    data_prefix_len = len(args.task_name) + 1
    file_num = len(file_list)
    for i in range(file_num):
        file_list[i] = file_list[i].split('/')[-1]
    file_list = sorted(file_list, key=lambda name: int(name[data_prefix_len:-4]))
    for file_name in file_list:
        if not send_appsrc_data(file_name, stream_name, stream_manager_api):
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

        logit_id = post_process(file_name, infer_result)
        if args.do_eval.lower() == "true":
            label_file = os.path.realpath(os.path.join(args.data_dir, "03_data", file_name))
            label_id = np.fromfile(label_file, np.int32)
            callback.update(logit_id, label_id)

    if args.do_eval.lower() == "true":
        print("==============================================================")
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
        print("==============================================================")
    scale = 1000.0
    print("Infer items sum:", file_num, "infer_total_time:", infer_total_time * scale, "ms")
    print("throughput:", file_num / infer_total_time, "bin/sec")

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    args = parse_args()
    callback = Accuracy()
    label_dic = read_label_file(os.path.realpath(args.label_file))
    run()
