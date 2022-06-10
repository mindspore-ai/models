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

"""ctc training"""

import glob
import os
import time
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector
from src.model_utils.config import config
from src.metric import LER
from mindspore import Tensor


def send_source_data(appsrc_id, filename, stream_name, stream_manager):
    tensor = np.fromfile(filename, dtype=np.float32)
    tensor = np.expand_dims(tensor, 0)
    if appsrc_id == 0:
        tensor = tensor.reshape(1, 1555, 39)
    else:
        tensor = tensor.reshape(1, 1555, 256)
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

def send_appsrc_data(data_dir, file_name, stream_name, stream_manager):
    feature = os.path.realpath(os.path.join(data_dir, "00_data", file_name))
    if not send_source_data(0, feature, stream_name, stream_manager):
        return False
    masks = os.path.realpath(os.path.join(data_dir, "01_data", file_name))
    if not send_source_data(1, masks, stream_name, stream_manager):
        return False
    return True


def post_process(metrics, file_name, infer_result, max_seq_length=1555):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 128.
    """

    print("==============================================================")
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    logits = res.reshape(config.max_sequence_length, -1, config.n_class)
    logits = Tensor(logits)

    labels = np.fromfile(os.path.join(config.label_dir, file_name), np.int32).reshape(config.test_batch_size, -1)
    labels = Tensor(labels)
    seq_len = np.fromfile(os.path.join(config.seqlen_dir, file_name), np.int32).reshape(-1)
    seq_len = Tensor(seq_len)

    metrics.update(logits, labels, seq_len)

def run():
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return
    with open("../data/config/ctcmodel.pipeline", 'rb') as f:
        pipeline = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    stream_name = b'ctcmodel'
    infer_total_time = 0
    file_list = glob.glob(os.path.join(config.data_url, "00_data", "*.bin"))
    metrics = LER(beam=config.beam)

    for file in file_list:
        file_name = file.split('/')[-1]
        if not send_appsrc_data(config.data_url, file_name, stream_name, stream_manager_api):
            return
        print("success"+file_name)
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        print("success infer result")
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        post_process(metrics, file_name, infer_result)
    print("Ler(310) is: ", metrics.eval())
    metrics.clear()

if __name__ == "__main__":
    run()
