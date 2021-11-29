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
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="gcn process")
    parser.add_argument("--dataset", type=str, default="cora", help="SDK infer pipeline")
    parser.add_argument('--data_dir', type=str, default='../data/input', help='Data path')
    parser.add_argument('--output_dir', type=str, default='./output', help='Data path')
    parser.add_argument("--pipeline", type=str, default="../utils/gcn_cora.pipeline", help="SDK infer pipeline")
    parser.add_argument('--data_adj', type=str, default="adjacency.txt")
    parser.add_argument("--data_feature", type=str, default="feature.txt")
    parser.add_argument('--data_label', type=str, default="label_onehot.txt")
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    args_opt = parser.parse_args()
    return args_opt

args = parse_args()

if args.dataset == "cora":
    Node_num = 2708
    Feature_dim = 1433
    Class_num = 7
    args.pipeline = "../data/config/gcn_cora.pipeline"
elif args.dataset == "citeseer":
    Node_num = 3312
    Feature_dim = 3703
    Class_num = 6
    args.pipeline = "../data/config/gcn_citeseer.pipeline"

adj_shape = [1, Node_num*Node_num]
feature_shape = [1, Node_num*Feature_dim]

def send_source_data(appsrc_id, filename, stream_name, stream_manager, shape, tp):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensors = (np.loadtxt(os.path.join(args.data_dir, args.dataset, filename), dtype=tp)).astype(np.float32)
    tensors = tensors.reshape(shape[0], shape[1])
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    data_input = MxDataInput()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in shape:
        tensor_vec.tensorShape.append(i)
    print(filename + " shape :", tensor_vec.tensorShape)
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

def send_appsrc_data(appsrc_id, file_name, stream_name, stream_manager, shape, tp):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
        bool: send data success or not
    """
    if not send_source_data(appsrc_id, file_name, stream_name, stream_manager, shape, tp):
        return False
    return True

def accurate(label, preds):
    """Accuracy with masking."""
    preds = preds.astype(np.float32)
    correct_prediction = np.equal(np.argmax(preds, axis=1), np.argmax(label, axis=1))
    accuracy_all = correct_prediction.astype(np.float32)
    mask = np.zeros([len(preds)]).astype(np.float32)
    mask[len(preds) - args.test_nodes_num:len(preds)] = 1
    mask = mask.astype(np.float32)
    mask_reduce = np.mean(mask)
    mask = mask / mask_reduce
    accuracy_all *= mask
    return np.mean(accuracy_all)

def post_process(infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 128.
    """
    def w2txt(file, data):
        s = "Infer labels:\n"
        for i in range(len(data)):
            s = s + "node %i : %i"%(i, data[i])
            s = s + "\n"
        with open(file, "w") as f:
            f.write(s)
    # get the infer result
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)

    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16)
    res = res.reshape((Node_num, Class_num))

    label = np.loadtxt(os.path.join(args.data_dir, args.dataset, args.data_label), dtype=np.int32)
    label = label.reshape((Node_num, Class_num))

    pred_label = np.argmax(res, axis=1)
    acc = accurate(label, res)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    w2txt(file=os.path.join(args.output_dir, args.dataset+"_predict_label.txt"), data=pred_label)
    print('============================  Infer Result ============================')
    print("Pred_label  label:{}".format(pred_label))
    print("Infer acc:%f"%(acc))
    print('=======================================================================')

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

    stream_name = b'gcn'
    infer_total_time = 0

    if not send_appsrc_data(0, args.data_adj, stream_name, stream_manager_api, adj_shape, np.float64):
        return

    if not send_appsrc_data(1, args.data_feature, stream_name, stream_manager_api, feature_shape, np.float32):
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

    post_process(infer_result)
    print("Infer cost total time: {:.6f} sec.".format(infer_total_time))
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
