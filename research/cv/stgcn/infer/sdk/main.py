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
"""run sdk"""
import sys
import math
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    """
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
    return ret

def run():
    """
    read pipeline and do infer
    """
    if len(sys.argv) == 4:
        dir_name = sys.argv[1]
        res_dir_name = sys.argv[2]
        n_pred = int(sys.argv[3])
    else:
        print("Please enter Dataset path| Inference result path "
              "such as ../data ./result 9")
        exit(1)
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("./pipeline/stgcn.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    # Construct the input of the stream

    n_his = 12
    zscore = preprocessing.StandardScaler()

    df = pd.read_csv(dir_name, header=None)
    data_col = df.shape[0]
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    dataset = df[len_train + len_val:]

    zscore.fit(df[: len_train])
    dataset = zscore.transform(dataset)

    n_vertex = dataset.shape[1]
    len_record = len(dataset)
    num = len_record - n_his - n_pred

    x = np.zeros([num, 1, n_his, n_vertex], np.float32)
    y = np.zeros([num, n_vertex], np.float32)

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = dataset[head: tail].reshape(1, n_his, n_vertex)
        y[i] = dataset[tail + n_pred - 1]

    labels = []
    predcitions = []
    stream_name = b'im_stgcn'
    #start infer
    for i in range(num):
        inPluginId = 0
        tensor = np.expand_dims(x[i], axis=0)
        uniqueId = send_source_data(0, tensor, stream_name, stream_manager_api)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            return

        # Obtain the inference result by specifying stream_name and uniqueId.
        start_time = datetime.datetime.now()

        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, inPluginId, keyVec)

        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))

        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        # get infer result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        # convert the inference result to Numpy array
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

        labels.append(zscore.inverse_transform(np.expand_dims(y[i], axis=0)).reshape(-1))
        predcitions.append(zscore.inverse_transform(np.expand_dims(res, axis=0)).reshape(-1))

    np.savetxt(res_dir_name+'labels.txt', np.array(labels))
    np.savetxt(res_dir_name+'predcitions.txt', np.array(predcitions))

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    run()
