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

import argparse
import os
from datetime import datetime
import pickle
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

class DatasetGenerator:
    def __init__(self, data_dir, drop=True):
        self.data = []
        self.label = []
        filelist = os.path.join(data_dir, "fea.lst")
        labellist = os.path.join(data_dir, "label.lst")
        with open(filelist, 'r') as fp:
            for fpa in fp:
                self.data.append(os.path.join(data_dir, fpa.strip()))
        with open(labellist, 'r') as fp:
            for lab in fp:
                self.label.append(os.path.join(data_dir, lab.strip()))
        if drop:
            self.data.pop()
            self.label.pop()
        print("dataset init ok, total len:", len(self.data))

    def __getitem__(self, ind):
        npdata = np.load(self.data[ind])
        nplabel = np.load(self.label[ind]).tolist()
        return npdata, nplabel[0]

    def __len__(self):
        return len(self.data)

def inference(input_tensor):
    tensor_bytes = input_tensor.tobytes()
    in_plugin_id = 0
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    dataInput = MxDataInput()
    dataInput.data = tensor_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for t in input_tensor.shape:
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
    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
    if infer_result.size() == 0:
        print("inferResult is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
        exit()
    # get infer result
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    # convert the inference result to Numpy array
    out = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_path', type=str, default='../data/config/ecapa_tdnn.pipeline')
    parser.add_argument('--eval_data_path', type=str, default='../data/feat_eval/')
    parser.add_argument('--output_path', type=str, default='../output/')
    parser.add_argument('--npy_path', type=str, default='../npy/')
    hparams = parser.parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(hparams.pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
    if not os.path.exists(hparams.npy_path):
        os.makedirs(hparams.npy_path)
    stream_name = b'ecapa_tdnn'
    eval_data_path = hparams.eval_data_path
    dataset_enroll = DatasetGenerator(eval_data_path, False)
    steps_per_epoch_enroll = len(dataset_enroll)
    print("size of enroll, test:", steps_per_epoch_enroll)
    fpath = os.path.join(hparams.npy_path, f'enroll_dict_bleeched.npy')
    files_len = len(os.listdir(hparams.eval_data_path))
    data = {}
    enroll_dict = dict()
    for index in range(0, 50000):
        if index >= len(dataset_enroll):
            exit()
        batchdata = dataset_enroll[index][0][:, :301, :]
        if index % 1000 == 0:
            print(f"{datetime.now()}, iter-{index}")
        embs = inference(batchdata)
        for index1 in range(0, 1):
            enroll_dict1 = dict()
            enroll_dict1[dataset_enroll[index][1]] = embs.copy()  #返回具有从改数组复制的值的numpy.ndarray对象
            with open(hparams.output_path+str(index)+'.txt', 'w') as f_write:
                f_write.write(str(enroll_dict1))
        enroll_dict[dataset_enroll[index][1]] = embs.copy()
        pickle.dump(enroll_dict, open(fpath, "wb"))

    # destroy streams
    stream_manager_api.DestroyAllStreams()
    