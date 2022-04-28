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
""" Link Prediction Evaluation """
import os
import numpy as np
from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

class KGEModel():
    """
    Generate sorted candidate entity id and positive sample.

    Args:
        network (nn.Cell): Trained model with entity embedding and relation embedding.
        mode (str): which negative sample mode ('head-mode' or 'tail-mode').

    Returns:
        argsort: entity id sorted by score
        positive_arg: positive sample entity id

    """
    def __init__(self, network_pipeline_, mode=b'head-mode'):
        self.network_pipeline = network_pipeline_
        self.mode = mode
        self.stream_manager = StreamManagerApi()
        ret = self.stream_manager.InitManager()
        if ret != 0:
            print("Failed to init Stream manager, ret=%s" % str(ret))
            exit()

        # create streams by pipeline config file
        with open(self.network_pipeline, 'rb') as f:
            pipeline = f.read()
        ret = self.stream_manager.CreateMultipleStreams(pipeline)
        if ret != 0:
            print("Failed to create Stream, ret=%s" % str(ret))
            exit()

    def infer(self, positive_sample, negative_sample, filter_bias):
        """
        infer
        """
        data_input1 = MxDataInput()
        print('positive_sample')
        print(positive_sample)
        data_input1.data = positive_sample.tobytes()
        tensorPackageList1 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage1 = tensorPackageList1.tensorPackageVec.add()
        tensorVec1 = tensorPackage1.tensorVec.add()
        tensorVec1.deviceId = 0
        tensorVec1.memType = 0
        for t in positive_sample.shape:
            tensorVec1.tensorShape.append(t)
        tensorVec1.dataStr = data_input1.data
        tensorVec1.tensorDataSize = len(positive_sample.tobytes())
        protobufVec1 = InProtobufVector()
        protobuf1 = MxProtobufIn()
        protobuf1.key = b'appsrc0'
        protobuf1.type = b'MxTools.MxpiTensorPackageList'
        protobuf1.protobuf = tensorPackageList1.SerializeToString()
        protobufVec1.push_back(protobuf1)


        self.stream_manager.SendProtobuf(self.mode, b'appsrc0', protobufVec1)

        # Construct the input of the stream
        data_input2 = MxDataInput()
        print('negative_sample')
        print(negative_sample)
        data_input2.data = negative_sample.tobytes()
        tensorPackageList2 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage2 = tensorPackageList2.tensorPackageVec.add()
        tensorVec2 = tensorPackage2.tensorVec.add()
        tensorVec2.deviceId = 0
        tensorVec2.memType = 0
        for t in negative_sample.shape:
            tensorVec2.tensorShape.append(t)
        tensorVec2.dataStr = data_input2.data
        tensorVec2.tensorDataSize = len(negative_sample.tobytes())
        protobufVec2 = InProtobufVector()
        protobuf2 = MxProtobufIn()
        protobuf2.key = b'appsrc1'
        protobuf2.type = b'MxTools.MxpiTensorPackageList'
        protobuf2.protobuf = tensorPackageList2.SerializeToString()
        protobufVec2.push_back(protobuf2)


        self.stream_manager.SendProtobuf(self.mode, b'appsrc1', protobufVec2)

        # Construct the input of the stream
        data_input3 = MxDataInput()
        print('filter_bias')
        print(filter_bias)
        data_input3.data = filter_bias.tobytes()
        tensorPackageList3 = MxpiDataType.MxpiTensorPackageList()
        tensorPackage3 = tensorPackageList3.tensorPackageVec.add()
        tensorVec3 = tensorPackage3.tensorVec.add()
        tensorVec3.deviceId = 0
        tensorVec3.memType = 0
        for t in filter_bias.shape:
            tensorVec3.tensorShape.append(t)
        tensorVec3.dataStr = data_input3.data
        tensorVec3.tensorDataSize = len(filter_bias.tobytes())
        protobufVec3 = InProtobufVector()
        protobuf3 = MxProtobufIn()
        protobuf3.key = b'appsrc2'
        protobuf3.type = b'MxTools.MxpiTensorPackageList'
        protobuf3.protobuf = tensorPackageList3.SerializeToString()
        protobufVec3.push_back(protobuf3)


        self.stream_manager.SendProtobuf(self.mode, b'appsrc2', protobufVec3)

        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        infer_result = self.stream_manager.GetProtobuf(self.mode, 0, keyVec)
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
        argsort = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32).reshape(1, -1)
        positive_arg = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.int32).reshape(1, -1)

        return argsort, positive_arg

    def construct(self, positive_sample, negative_sample, filter_bias):
        """ Sort candidate entity id and positive sample entity id. """

        score = self.infer(positive_sample, negative_sample, filter_bias)

        if self.mode == b'head-mode':

            positive_arg = positive_sample[:, 0]
        else:

            positive_arg = positive_sample[:, 2]

        argsort = (0 - score).argsort()

        return argsort, positive_arg


class EvalKGEMetric():
    """
    Calculate metrics.

    Args:
        network (nn.Cell): Trained model with entity embedding and relation embedding.
        mode (str): which negative sample mode ('head-mode' or 'tail-mode').

    Returns:
        log (list): contain metrics of each triple

    """

    def __init__(self, network_pipeline__, mode=b'head-mode'):
        self.mode = mode
        self.kgemodel = KGEModel(network_pipeline_=network_pipeline__, mode=self.mode)
    def construct(self, positive_sample, negative_sample, filter_bias):
        """Calculate metrics"""
        positive_sample = positive_sample.reshape(1, -1).astype(np.int32)
        negative_sample = negative_sample.reshape(1, -1).astype(np.int32)
        filter_bias = filter_bias.reshape(1, -1).astype(np.float32)

        batch_size = positive_sample.shape[0]
        argsort, positive_arg = self.kgemodel.infer(positive_sample, negative_sample, filter_bias)

        log = []
        for i in range(batch_size):
            ranking = np.where(argsort[i, :] == positive_arg[i])[0][0]
            ranking = 1 + ranking
            log.append({
                'MRR': 1.0 / ranking,
                'MR': ranking,
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0,
            })
        return log, argsort, positive_arg


def eval_kge(network_pipeline, data_pth):
    """ Link Prediction Task for Knowledge Graph Embedding Model """
    positive_sample_head = np.loadtxt(data_pth + '/positive_sample_head.txt', delimiter=' ')
    negative_sample_head = np.loadtxt(data_pth + '/negative_sample_head.txt', delimiter=' ')
    filter_bias_head = np.loadtxt(data_pth + '/filter_bias_head.txt', delimiter=' ')

    positive_sample_tail = np.loadtxt(data_pth + '/positive_sample_tail.txt', delimiter=' ')
    negative_sample_tail = np.loadtxt(data_pth + '/negative_sample_tail.txt', delimiter=' ')
    filter_bias_tail = np.loadtxt(data_pth + '/filter_bias_tail.txt', delimiter=' ')

    print('positive_sample')
    print(positive_sample_head)
    print('negative_sample')
    print(negative_sample_head)
    print('filter_bias')
    print(filter_bias_head)

    logs = []
    eval_model_head = EvalKGEMetric(network_pipeline__=network_pipeline, mode=b'head-mode')
    eval_model_tail = EvalKGEMetric(network_pipeline__=network_pipeline, mode=b'tail-mode')

    argsort_list = []
    positive_arg_list = []
    for test_data in zip(positive_sample_head, negative_sample_head, filter_bias_head):
        log_head, argsort, positive_arg = eval_model_head.construct(test_data[0], test_data[1], test_data[2])
        argsort_list.append(argsort[0])
        positive_arg_list.append(positive_arg[0])
        logs += log_head
    for test_data in zip(positive_sample_tail, negative_sample_tail, filter_bias_tail):
        log_tail, argsort, positive_arg = eval_model_tail.construct(test_data[0], test_data[1], test_data[2])
        argsort_list.append(argsort[0])
        positive_arg_list.append(positive_arg[0])
        logs += log_tail

    argsort_list = np.array(argsort_list)
    positive_arg_list = np.array(positive_arg_list)
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savetxt('results/argsort.txt', argsort_list, fmt='%i')
    np.savetxt('results/positive_arg.txt', positive_arg_list, fmt='%i')

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    print(metrics)


if __name__ == '__main__':
    net_pipeline = '../data/config/rotate.pipeline'
    data_path = '../data/wn18rr'
    eval_kge(net_pipeline, data_path)
