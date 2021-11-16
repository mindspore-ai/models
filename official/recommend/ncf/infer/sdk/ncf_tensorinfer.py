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
Use SDK to infer running in docker.
"""
import os
import glob
import time
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import InProtobufVector
from StreamManagerApi import StringVector
import util.constants as rconst

def hit(target, pred_items):
    """compute hit"""
    if target in pred_items:
        return 1
    return 0

def ndcg_function(target, pred_items):
    """compute ndcg"""
    if target in pred_items:
        rank = pred_items.index(target)
        return np.reciprocal(np.log2(rank + 2))
    return 0

def run():
    """MindX SDK inference"""
    # init stream manager
    pipeline_path = "../data/config/NCF.pipeline"
    streamName = b'ncf'

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    eval_users_per_batch = int(
                rconst.BATCH_SIZE // (1 + rconst.NUM_EVAL_NEGATIVES))
    hr_sum = 0.0
    hr_num = 0
    ndcgr_sum = 0.0
    ndcgr_num = 0
    infer_total_time = 0.0
    i_batch = 0

    users_path = "../data/input/tensor_0"
    items_path = "../data/input/tensor_1"
    masks_path = "../data/input/tensor_2"
    # input file list
    file_list = glob.glob(os.path.join(os.path.realpath(users_path), "*.txt"))
    for input_path in file_list:
        # load inputs
        file_name = input_path.split('/')[-1]

        users_file = os.path.join(users_path, file_name)
        items_file = os.path.join(items_path, file_name)
        masks_file = os.path.join(masks_path, file_name)

        users = np.fromfile(users_file, dtype=np.int32).reshape(1, rconst.BATCH_SIZE)
        items = np.fromfile(items_file, dtype=np.int32).reshape(1, rconst.BATCH_SIZE)
        masks = np.fromfile(masks_file, dtype=np.float32).reshape(1, rconst.BATCH_SIZE)
        tensors = [users, items, masks]

        print("=" * 20 + "Batch_{} Eval Begin!".format(i_batch) + "=" * 20)
        inPluginId = 0
        for tensor in tensors:
            tensorPackageList = MxpiDataType.MxpiTensorPackageList()
            tensorPackage = tensorPackageList.tensorPackageVec.add()
            array_bytes = tensor.tobytes()
            tensorVec = tensorPackage.tensorVec.add()
            tensorVec.deviceId = 0
            tensorVec.memType = 0
            for i in tensor.shape:
                tensorVec.tensorShape.append(i)
            tensorVec.dataStr = array_bytes
            tensorVec.tensorDataSize = len(array_bytes)

            key = "appsrc{}".format(inPluginId).encode('utf-8')
            protobufVec = InProtobufVector()
            protobuf = MxProtobufIn()
            protobuf.key = key
            protobuf.type = b'MxTools.MxpiTensorPackageList'
            protobuf.protobuf = tensorPackageList.SerializeToString()
            protobufVec.push_back(protobuf)
            uniqueId = streamManagerApi.SendProtobuf(streamName, inPluginId, protobufVec)
            inPluginId += 1
            if uniqueId < 0:
                print("Failed to send data to stream")
                exit()

        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        inferResult = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        infer_total_time += time.time() - start_time

        if inferResult.size() == 0:
            print("inferResult is null")
            exit()

        if inferResult[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                inferResult[0].errorCode))
            exit()

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(inferResult[0].messageBuf)
        indices = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)
        output_items = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.int32)
        metric_weights = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr, dtype=bool)

        # postprocess
        hr = []
        ndcg = []
        indices = np.reshape(indices, (eval_users_per_batch, rconst.TOP_K))
        output_items = np.reshape(output_items, (eval_users_per_batch, 1 + rconst.NUM_EVAL_NEGATIVES))
        for index in range(eval_users_per_batch):
            if metric_weights[index]:
                recommends = output_items[index][indices[index]].tolist()
                gt_item = output_items[index].tolist()[-1]
                hr.append(hit(gt_item, recommends))
                ndcg.append(ndcg_function(gt_item, recommends))

        hr_sum += np.sum(hr)
        ndcgr_sum += np.sum(ndcg)
        hr_num += len(hr)
        ndcgr_num += len(ndcg)
        print("EvalCallBack: HR = {}, NDCG = {}".format(np.mean(hr), np.mean(ndcg)))
        print("=" * 20 + "Batch_{} Eval Finish!".format(i_batch) + "=" * 20)
        i_batch += 1

    print("=" * 20 + "Average Eval Begin!" + "=" * 20)
    print("average HR = {:.6f}, average NDCG = {:.6f}".format(np.mean(hr_sum/hr_num), np.mean(ndcgr_sum/ndcgr_num)))
    print("=" * 20 + "Average Eval Finish!" + "=" * 20)
    print("Infer {} batches, cost total time: {:.6f} sec.".format(len(file_list), infer_total_time))
    print("Average cost {:.6f} sec per batch".format(infer_total_time/len(file_list)))
    # destroy streams
    streamManagerApi.DestroyAllStreams()

if __name__ == '__main__':
    run()
