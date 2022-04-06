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
import os

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, \
    InProtobufVector, MxProtobufIn, StringVector

from util.dataset import create_eval_dataset, EvalNews, EvalUsers, EvalCandidateNews, MINDPreprocess
from util.config import config
from util.utils import NAMLMetric


def news_process(streamapi, news_pipeline_path, news_data):
    """Perform news reasoning
         Returns:
         news_dict = {}
    """
    streamName = b'news_pipeline'
    with open(news_pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    streamapi.CreateMultipleStreams(pipelineStr)
    iterator = news_data.create_dict_iterator(output_numpy=True)
    tensors = []
    news_dict = {}
    news_dataset_size = news_data.get_dataset_size()
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')

    for count, data in enumerate(iterator):
        tensors.clear()
        tensors.append(data["category"])
        tensors.append(data["subcategory"])
        tensors.append(data["title"])
        tensors.append(data["abstract"])
        if not send_data(tensors, streamName, streamapi):
            exit()
        ret = streamapi.GetProtobuf(streamName, 0, key_vec)
        if ret.size() == 0:
            print("inferResult is null")
            exit()
        if ret[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % ret[0].errorCode)
            exit()
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(ret[0].messageBuf)
        news_vec = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        for nid in enumerate(data["news_id"]):
            news_dict[str(nid[1][0])] = news_vec
        print(f"===Generate News vector==== [ {count} / {news_dataset_size} ]", end='\r')
    print(f"===Generate News vector==== [ {news_dataset_size} / {news_dataset_size} ]")
    return news_dict


def user_process(streamapi, user_pipeline_path, user_data, news_process_ret):
    """Perform user reasoning
         Returns:
         user_dict = {}
    """
    streamName = b'user_pipeline'
    with open(user_pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    streamapi.CreateMultipleStreams(pipelineStr)
    user_data_size = user_data.get_dataset_size()
    tensors = []
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer1')
    iterator = user_data.create_dict_iterator(output_numpy=True)
    user_dict = {}
    for count, data in enumerate(iterator):
        tensors.clear()
        browsed_news = []
        for news in data["history"]:
            news_list = []
            for nid in news:
                news_list.append(news_process_ret[str(nid[0])])
            browsed_news.append(np.array(news_list))
        browsed_news = np.array(browsed_news)
        tensors.append(browsed_news)
        if not send_data(tensors, streamName, streamapi):
            exit()
        ret = streamapi.GetProtobuf(streamName, 0, key_vec)
        if ret.size() == 0:
            print("inferResult is null")
            exit()
        if ret[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % ret[0].errorCode)
            exit()
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(ret[0].messageBuf)
        user_vec = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        for uid in enumerate(data["uid"]):
            user_dict[str(uid[1])] = user_vec
        print(f"===Generate Users vector==== [ {count} / {user_data_size} ]", end='\r')
    print(f"===Generate Users vector==== [ {user_data_size} / {user_data_size} ]")
    streamapi.DestroyAllStreams()
    return user_dict


def create_dataset(mindpreprocess, datatype, batch_size):
    """create_dataset"""
    dataset = create_eval_dataset(mindpreprocess, datatype, batch_size)
    return dataset


def send_data(tensors, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

     Returns:
        bool: send data success or not
    """

    inPluginId = 0
    for tensor in tensors:
        tensorPackageList = MxpiDataType.MxpiTensorPackageList()
        tensorPackage = tensorPackageList.tensorPackageVec.add()
        array_bytes = tensor.tobytes()
        dataInput = MxDataInput()
        dataInput.data = array_bytes
        tensorVec = tensorPackage.tensorVec.add()
        tensorVec.deviceId = 0
        tensorVec.memType = 0
        for i in tensor.shape:
            tensorVec.tensorShape.append(i)
        tensorVec.dataStr = dataInput.data
        tensorVec.tensorDataSize = len(array_bytes)
        key = "appsrc{}".format(inPluginId).encode('utf-8')
        protobufVec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensorPackageList.SerializeToString()
        protobufVec.push_back(protobuf)
        ret = stream_manager.SendProtobuf(stream_name, inPluginId, protobufVec)
        inPluginId += 1
        if ret != 0:
            print("Failed to send data to stream.")
            return False
    return True


def run_process():
    """run naml model SDK process"""
    if config.neg_sample == 4:
        config.neg_sample = -1
    if config.batch_size != 1:
        config.batch_size = 1
    config.embedding_file = os.path.join(config.dataset_path, config.embedding_file)
    config.word_dict_path = os.path.join(config.dataset_path, config.word_dict_path)
    config.category_dict_path = os.path.join(config.dataset_path, config.category_dict_path)
    config.subcategory_dict_path = os.path.join(config.dataset_path, config.subcategory_dict_path)
    config.uid2index_path = os.path.join(config.dataset_path, config.uid2index_path)
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    news_pi_path = "../data/config/news.pipeline"
    user_pi_path = "../data/config/user.pipeline"
    mindpreprocess = MINDPreprocess(vars(config), dataset_path=os.path.join(config.dataset_path,
                                                                            "MIND{}_dev".format(config.dataset)))
    news_data = create_dataset(mindpreprocess, EvalNews, 1)
    user_data = create_dataset(mindpreprocess, EvalUsers, 1)
    news_dict = news_process(streamManagerApi, news_pi_path, news_data)
    print("start to user_process")
    user_dict = user_process(streamManagerApi, user_pi_path, user_data, news_dict)
    print("start to metric")
    eval_data = create_dataset(mindpreprocess, EvalCandidateNews, 1)
    dataset_size = eval_data.get_dataset_size()
    iterator = eval_data.create_dict_iterator(output_numpy=True)
    metric = NAMLMetric()
    for count, data in enumerate(iterator):
        pred = np.dot(np.stack([news_dict[str(nid)] for nid in data["candidate_nid"]], axis=0),
                      user_dict[str(data["uid"])])
        metric.update(pred, data["labels"])
        print(f"===Click Prediction==== [ {count} / {dataset_size} ]", end='\r')
    print(f"===Click Prediction==== [ {dataset_size} / {dataset_size} ]")
    metric.eval()

if __name__ == '__main__':
    run_process()
