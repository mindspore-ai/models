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
sample script of gpt2 infer using SDK run in docker
"""

import os
import argparse
import math
import numpy as np

from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

batch_size = 1
seq_length = 1024
vocab_size = 50257

def cross_entropy_calculation(logits, label_ids, input_mask):
    '''
    Calculate cross entropy with mask.
    '''
    label_ids = np.reshape(label_ids, (-1,))  # label_ids [batch * (seq_length-1)]
    one_hot_labels = np.eye(vocab_size)[label_ids]
    per_example_loss = np.negative(np.sum(one_hot_labels * logits, axis=1))

    # for PPL calculation in evaluation
    input_mask = np.reshape(input_mask, (-1,))
    input_mask = input_mask.astype(np.float32)
    valid_loss_sum = np.sum(input_mask * per_example_loss)
    valid_element_sum = np.sum(input_mask) + 1e-5
    loss = valid_loss_sum / valid_element_sum
    return_value = loss

    return return_value

def create_protobufVec(data, key):
    """
    Create protobufVec
    """
    data_input = MxDataInput()
    data_input.data = data.tobytes()
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for t in data.shape:
        tensorVec.tensorShape.append(t)
    tensorVec.dataStr = data_input.data
    tensorVec.tensorDataSize = len(data.tobytes())
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    return protobufVec

def infer(stream_manager, stream_name, input_ids, input_masks, label_ids):
    """
    SDK infer

    Args:
        stream_manager: the object of stream.
        stream_name: the name of stream.
        input_ids: the indices of input sequence tokens in the vocabulary. shape is [batch_size, seq_length].
        input_mask: input sentences padding mask, where 0 indicates padding position. shape is [batch_size, seq_length].
        label_ids: the indices of input sequence tokens in the vocabulary. shape is [batch_size, seq_length].

    return:
        logit_id: shape is [batch_size, seq_length, vocab_size]
    """
    ret = stream_manager.SendProtobuf(stream_name, b'appsrc0', create_protobufVec(input_ids, b'appsrc0'))
    if ret != 0:
        print('Fialed sendprotobuf for inputs_id. ret=', ret)
        return None
    ret = stream_manager.SendProtobuf(stream_name, b'appsrc1', create_protobufVec(input_masks, b'appsrc1'))
    if ret != 0:
        print('Fialed sendprotobuf for input_masks. ret=', ret)
        return None
    ret = stream_manager.SendProtobuf(stream_name, b'appsrc2', create_protobufVec(label_ids, b'appsrc2'))
    if ret != 0:
        print('Fialed sendprotobuf for label_ids. ret=', ret)
        return None

    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager.GetProtobuf(stream_name, 0, keyVec)
    # check error
    if infer_result.size() == 0:
        print("InferResult is null")
        return None
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        return None
    print("key:" + str(infer_result[0].messageName))
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    # convert the inference result to Numpy array
    logit_id = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                             dtype=np.float32).reshape(batch_size,
                                                       seq_length,
                                                       vocab_size)

    return logit_id

def run_gpt2(data_path, network_pipeline, logs_path):
    """
    run gpt2 task

    Args:
        datapath: the path of input_ids.txt, input_mask.txt, label_ids.txt.
        network_pipeline: the path of pipeline.
    """
    input_ids = np.loadtxt(os.path.realpath(data_path + 'input_ids.txt'), dtype=int, delimiter=' ').astype(np.int32)
    input_mask = np.loadtxt(os.path.realpath(data_path + 'input_mask.txt'), dtype=int, delimiter=' ').astype(np.int32)
    label_ids = np.loadtxt(os.path.realpath(data_path + 'label_ids.txt'), dtype=int, delimiter=' ').astype(np.int32)

    data_len = input_ids.shape[0]

    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(network_pipeline), 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
    if not os.path.exists('results'):
        os.makedirs('results')
    f = open(logs_path + "score.txt", 'w')

    num_data = 1
    total_loss = 0.0
    avg_loss = 0.0
    print("====================    Start inferring   ==================")
    print(" | Dataset path: {}".format(data_path))
    for i in range(data_len):
        output = infer(stream_manager, b"im_gpt2", input_ids[i:i+1].reshape(1, -1),
                       input_mask[i:i+1].reshape(1, -1), label_ids[i:i+1].reshape(1, -1))
        output = output[::, :-1, ::]
        loss_new = cross_entropy_calculation(output.reshape(seq_length - 1, vocab_size),
                                             label_ids[i:i+1][::, 1:],
                                             input_mask[i:i+1][::, 1:])
        total_loss += float(loss_new)
        avg_loss = total_loss / num_data
        print(" | Current Loss: {:.6f}".format(float(loss_new)))
        print(" | Current PLL: {}\n\n".format(math.exp(float(loss_new))))
        num_data += 1
        f.write(str(math.exp(float(loss_new))) + '\n')
        f.flush()
    print(" | Average Loss: {:.6f}".format(avg_loss))
    print(" | Average PLL: {}\n\n".format(math.exp(avg_loss)))
    print("====================    inferring Finished   ================")
    f.close()
    stream_manager.DestroyAllStreams()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" SDK infer language modelings task")
    parser.add_argument("--data_dir", type=str, default="./infer/data/data/",
                        help="Load the data file path for infer.")
    parser.add_argument("--pipeline_path", type=str, default="./infer/data/config/gpt2.pipeline",
                        help="Load the pipeline file path for infer.")
    parser.add_argument("--logs_dir", type=str, default="",
                        help="Save the logs file.")
    args_opt = parser.parse_args()
    data_dir = args_opt.data_dir
    pipeline_path = args_opt.pipeline_path
    logs_dir = args_opt.logs_dir
    run_gpt2(data_dir, pipeline_path, logs_dir)
