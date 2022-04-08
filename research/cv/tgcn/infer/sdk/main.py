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

import datetime
import numpy as np
from sklearn import metrics
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput


def accuracy(preds, targets):
    """
    Calculate the accuracy between predictions and targets

    Args:
        preds(Tensor): predictions
        targets(Tensor): ground truth

    Returns:
        accuracy: defined as 1 - (norm(targets - preds) / norm(targets))
    """
    return 1 - np.linalg.norm(targets - preds) / np.linalg.norm(targets)


def r2(preds, targets):
    """
    Calculate R square between predictions and targets

    Args:
        preds(Tensor): predictions
        targets(Tensor): ground truth

    Returns:
        R square: coefficient of determination
    """
    return 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.sum(preds)) ** 2)


def explained_variance(preds, targets):
    """
    Calculate the explained variance between predictions and targets

    Args:
        preds(Tensor): predictions
        targets(Tensor): ground truth

    Returns:
        Var: explained variance
    """
    return 1 - (targets - preds).var() / targets.var()

def load_feat_matrix(path):
    feat = np.loadtxt(path, delimiter=',', skiprows=1)
    tmp_max_val = np.max(feat)
    return feat, tmp_max_val

def generate_dataset_np(feat, seq_len, pre_len, normalize=True):
    time_len = feat.shape[0]
    if normalize:
        tmp_max_val = np.max(feat)
        feat = feat / tmp_max_val
    train_size = int(time_len * 0.8)
    train_data = feat[0:train_size]
    eval_data = feat[train_size:time_len]
    train_inputs, train_targets, tmp_eval_inputs, tmp_eval_targets = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_inputs.append(np.array(train_data[i: i + seq_len]))
        train_targets.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(eval_data) - seq_len - pre_len):
        tmp_eval_inputs.append(np.array(eval_data[i: i + seq_len]))
        tmp_eval_targets.append(np.array(eval_data[i + seq_len: i + seq_len + pre_len]))
    return np.array(train_inputs), np.array(train_targets), np.array(tmp_eval_inputs), np.array(tmp_eval_targets)

def get_dataset(path):
    feat, tmp_max_val = load_feat_matrix(path)
    _, _, tmp_eval_inputs, tmp_eval_targets = generate_dataset_np(feat, 4, 1)
    return tmp_max_val, tmp_eval_inputs, tmp_eval_targets

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/tgcn.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    infer_total_time = 0
    data_path = '../data/input/SZ-taxi/feature.csv'

    # all_test_data = np.loadtxt(data_path)
    # all_test_label = np.loadtxt(label_path)
    # all_test_label = all_test_label.astype(np.int32)

    stream_name = b'im_tgcn'
    max_val, eval_inputs, eval_targets = get_dataset(data_path)
    print("eval input shape")
    print(eval_inputs.shape)
    print("eval target shape")
    print(eval_targets.shape)
    num = eval_inputs.shape[0]
    dataset = np.zeros([num, 1, 4, 156], np.float32)
    for idx in range(num):
        dataset[idx, :, :, :] = eval_inputs[idx].reshape(4, 156)
    bs = 64
    tot_output = []
    tot_rmse = 0
    tot_mae = 0
    tot_acc = 0
    tot_r2 = 0
    tot_var = 0
    for idx in range(num):
        tensor = dataset[idx]
        tensor_bytes = tensor.tobytes()
        in_plugin_id = 0
        tensorPackageList = MxpiDataType.MxpiTensorPackageList()
        tensorPackage = tensorPackageList.tensorPackageVec.add()
        dataInput = MxDataInput()
        dataInput.data = tensor_bytes
        tensorVec = tensorPackage.tensorVec.add()
        tensorVec.deviceId = 0
        tensorVec.memType = 0
        for t in tensor.shape:
            tensorVec.tensorShape.append(t)
        tensorVec.dataStr = dataInput.data
        tensorVec.tensorDataSize = len(tensor_bytes)
        # add feature data end
        key = "appsrc{}".format(in_plugin_id).encode('utf-8')
        protobufVec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensorPackageList.SerializeToString()
        protobufVec.push_back(protobuf)
        unique_id = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobufVec)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        start_time = datetime.datetime.now()
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
        if infer_result.size() == 0:
            print("inferResult is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()
        # get infer result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        # convert the inference result to Numpy array
        output = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        tmp_target = np.squeeze(eval_targets[idx], axis=0)
        tot_rmse += np.sqrt(metrics.mean_squared_error(tmp_target, output))
        tot_mae += metrics.mean_absolute_error(tmp_target, output)
        tot_acc += accuracy(tmp_target, output)
        tot_r2 += r2(tmp_target, output)
        tot_var += explained_variance(tmp_target, output)
        tot_output.append(output)
    with open('res.txt', 'w') as f:
        for output in tot_output:
            for x in output:
                f.write(('%.6f'%x)  + ' ')
            f.write('\n')

    print("=====Evaluation Results=====")
    print('RMSE:', '{:.6f}'.format(tot_rmse * max_val / num))
    print('MAE:', '{:.6f}'.format(tot_mae * max_val / num))
    print('Accuracy:', '{:.6f}'.format(tot_acc / num))
    print('R2:', '{:.6f}'.format(tot_r2 / num))
    print('Var:', '{:.6f}'.format(tot_var / num))
    print("============================")
    # destroy streams
    stream_manager_api.DestroyAllStreams()
