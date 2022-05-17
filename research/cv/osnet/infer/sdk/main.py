# coding=utf-8

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
# pylint: skip-file

import os
import sys
import time
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
config_path = CURRENT_DIR.rsplit('/', 2)[0]
sys.path.append(config_path)
from src.dataset import dataset_creator
from model_utils.config import config

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    Args:
        appsrc_id: an RGB image:the appsrc component number for SendProtobuf
        tensor: the tensor type of the input file
        stream_name: stream Name
        stream_manager:the StreamManagerApi
    Returns:
        bool: send data success or not
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
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    os.makedirs("./result")
    os.makedirs("./result/result")
    os.makedirs("./result/label")
    os.makedirs("./result/camlabel")
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("../data/config/OSNet.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
    _, query_dataset = dataset_creator(root='../data/data/', height=config.height, width=config.width,
                                       dataset=config.target, norm_mean=config.norm_mean,
                                       norm_std=config.norm_std, batch_size_test=config.batch_size_test,
                                       workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                       cuhk03_classic_split=config.cuhk03_classic_split, mode='query')
    _, gallery_dataset = dataset_creator(root='../data/data/', height=config.height,
                                         width=config.width, dataset=config.target,
                                         norm_mean=config.norm_mean, norm_std=config.norm_std,
                                         batch_size_test=config.batch_size_test, workers=config.workers,
                                         cuhk03_labeled=config.cuhk03_labeled,
                                         cuhk03_classic_split=config.cuhk03_classic_split,
                                         mode='gallery')
    def feature_extraction(eval_dataset, name):
        infer_total_time = 0
        for idx, data in enumerate(eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
            imgs, pids, camids = data['img'], data['pid'], data['camid']
            stream_name = b'OSNet'
            if not send_source_data(0, imgs, stream_name, stream_manager_api):
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
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            if name == 'query':
                file_name = './result/result/' + "query_" + str(config.target) + "_" + \
                            str(config.batch_size_test) + "_" + str(idx) + ".txt"
                label_file_path = './result/label/' + "query_" + str(config.target) + "_" + \
                                  str(config.batch_size_test) + "_" + str(idx) + ".bin"
                camlabel_file_path = './result/camlabel/' + "query_" + str(config.target) + "_" + \
                                     str(config.batch_size_test) + "_" + str(idx) + ".bin"
            else:
                file_name = './result/result/' + "gallery_" + str(config.target) + "_" + \
                             str(config.batch_size_test) + "_" + str(idx) + ".txt"
                label_file_path = './result/label/' + "gallery_" + str(config.target) + "_" + \
                                  str(config.batch_size_test) + "_" + str(idx) + ".bin"
                camlabel_file_path = './result/camlabel/' + "gallery_" + str(config.target) + "_" + \
                             str(config.batch_size_test) + "_" + str(idx) + ".bin"
            pids.tofile(label_file_path)
            camids.tofile(camlabel_file_path)
            res = list(res)
            f = open(file_name, 'w')
            for i in range(len(res)):
                f.writelines(str(res[i]) + '\n')
            f.close()
        return infer_total_time
    print('Extracting features from query set ...')
    time1 = feature_extraction(query_dataset, 'query')

    print('Extracting features from gallery set ...')
    time2 = feature_extraction(gallery_dataset, 'gallery')

    print("=======================================")
    print("The total time of inference is {} s".format(time1+time2))
    print("=======================================")
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
