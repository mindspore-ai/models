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
eval
"""

import os
import argparse
import time
import math
import heapq
from multiprocessing import Pool
import pickle as pkl
import numpy as np


import MxpiDataType_pb2 as MxpiDataType

from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bgcf process")
    parser.add_argument("--eval", type=str,
                        default="../../data/eval", help="eval file")
    parser.add_argument("--pipeline", type=str,
                        default="../../data/config/bgcf.pipeline", help="SDK infer pipeline")
    parser.add_argument("--infer", type=str,
                        default='infer', help="is infer or eval")
    args_opt = parser.parse_args()
    return args_opt


args = parse_args()

num_user = 7068
num_item = 3570

row_neighs = 40
gnew_neighs = 20

Ks = [5, 10, 20, 100]

with open(args.eval + '/test_inputs.pkl', 'rb') as file:
    test_inputs = pkl.load(file)
with open(args.eval + '/test_set.pkl', 'rb') as file:
    test_set = pkl.load(file)
with open(args.eval + '/train_set.pkl', 'rb') as file:
    train_set = pkl.load(file)
with open(args.eval + '/item_deg_dict.pkl', 'rb') as file:
    item_deg_dict = pkl.load(file)
with open(args.eval + '/item_full_set.pkl', 'rb') as file:
    item_full_set = pkl.load(file, encoding="...")

test_input = test_inputs[0]
users = test_input[0].reshape(1, num_user)
items = test_input[1].reshape(1, num_item)
neg_items = test_input[2].reshape(1, num_item)
u_test_neighs = test_input[3].reshape([1, num_user*row_neighs])
u_test_gnew_neighs = test_input[4].reshape([1, num_user*gnew_neighs])
i_test_neighs = test_input[5].reshape([1, num_item*row_neighs])
i_test_gnew_neighs = test_input[6].reshape([1, num_item*gnew_neighs])

def send_source_data(appsrc_id, filename, stream_name, stream_manager, shape, tp):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensors = (filename).astype(np.int32)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    data_input = MxDataInput()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in shape:
        tensor_vec.tensorShape.append(i)
    print(" shape :", tensor_vec.tensorShape)
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


def idcg_k(actual, k):
    """Calculates the ideal discounted cumulative gain at k"""
    res = sum([1.0 / math.log(i + 2, 2) for i in range(min(k, len(actual)))])
    return 1.0 if not res else res


def ndcg_k(actual, predicted, topk):
    """Calculates the normalized discounted cumulative gain at k"""
    idcg = idcg_k(actual, topk)
    res = 0

    dcg_k = sum([int(predicted[j] in set(actual)) / math.log(j + 2, 2)
                 for j in range(topk)])
    res += dcg_k / idcg
    return res


def recall_at_k_2(r, k, all_pos_num):
    """Calculates the recall at k"""
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def novelty_at_k(topk_items, item_degree_dict, num_user_t, k):
    """Calculate the novelty at k"""
    avg_nov = []
    for item in topk_items[:k]:
        avg_nov.append(-np.log2((item_degree_dict[item] + 1e-8) / num_user_t))
    return np.mean(avg_nov)


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks_t):
    """Return the n largest score from the item_score by heap algorithm"""
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks_t)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r, K_max_item_score


def get_performance(user_pos_test, r, K_max_item, item_degree_dict, num_user_t, Ks_t):
    """Wraps the model metrics"""
    recall, ndcg, novelty = [], [], []
    for K in Ks_t:
        recall.append(recall_at_k_2(r, K, len(user_pos_test)))
        ndcg.append(ndcg_k(user_pos_test, K_max_item, K))
        novelty.append(novelty_at_k(K_max_item, item_degree_dict, num_user_t, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg), 'nov': np.array(novelty)}


def test_one_user(x):
    """Calculate one user metrics"""
    rating = x[0]
    u = x[1]

    training_items = train_set[u]

    user_pos_test = test_set[u]

    all_items = set(range(num_item))

    test_items = list(all_items - set(training_items))

    r, k_max_items = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, k_max_items, item_deg_dict, num_user, Ks), \
        [k_max_items[:Ks[x]] for x in range(len(Ks))]


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

    stream_name = b'bgcf_gnn'
    infer_total_time = 0

    if not send_appsrc_data(0, users, stream_name, stream_manager_api, users.shape, np.int64):
        return
    if not send_appsrc_data(1, items, stream_name, stream_manager_api, items.shape, np.int32):
        return
    if not send_appsrc_data(2, neg_items, stream_name, stream_manager_api, neg_items.shape, np.int32):
        return
    if not send_appsrc_data(3, u_test_neighs, stream_name, stream_manager_api, u_test_neighs.shape, np.int32):
        return
    if not send_appsrc_data(4, u_test_gnew_neighs, stream_name, stream_manager_api, u_test_gnew_neighs.shape, np.int32):
        return
    if not send_appsrc_data(5, i_test_neighs, stream_name, stream_manager_api, i_test_neighs.shape, np.int32):
        return
    if not send_appsrc_data(6, i_test_gnew_neighs, stream_name, stream_manager_api, i_test_gnew_neighs.shape, np.int32):
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
    user_rep = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16).reshape(7068, 192)
    print(user_rep.shape)

    item_rep = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float16).reshape(3570, 192)
    print(item_rep.shape)

    np.save('./output/user_rep.npy', user_rep)
    np.save('./output/item_rep.npy', item_rep)

    stream_manager_api.DestroyAllStreams()

def eval_sdk():
    """
    eval
    """
    user_rep = np.load('./output/user_rep.npy')
    item_rep = np.load('./output/item_rep.npy')

    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'nov': np.zeros(len(Ks))}
    pool = Pool(8)
    user_indexes = np.arange(num_user)

    rating_preds = user_rep @ item_rep.transpose()
    user_rating_uid = zip(rating_preds, user_indexes)
    all_result = pool.map(test_one_user, user_rating_uid)

    top20 = []

    for re in all_result:
        result['recall'] += re[0]['recall'] / num_user
        result['ndcg'] += re[0]['ndcg'] / num_user
        result['nov'] += re[0]['nov'] / num_user
        top20.append(re[1][2])

    pool.close()

    sedp = [[] for i in range(len(Ks) - 1)]

    num_all_links = np.sum([len(x) for x in item_full_set])

    for k in range(len(Ks) - 1):
        for u in range(num_user):
            diff = []
            pred_items_at_k = all_result[u][1][k]
            for item in pred_items_at_k:
                if item in test_set[u]:
                    avg_prob_all_user = len(
                        item_full_set[item]) / num_all_links
                    diff.append(max((Ks[k] - pred_items_at_k.index(item) - 1)
                                    / (Ks[k] - 1) - avg_prob_all_user, 0))
            one_user_sedp = sum(diff) / Ks[k]
            sedp[k].append(one_user_sedp)

    sedp = np.array(sedp).mean(1)

    test_recall_bgcf, test_ndcg_bgcf, \
        test_sedp, test_nov = result['recall'].tolist(), result['ndcg'].tolist(), \
        [sedp[1], sedp[2]], result['nov'].tolist()
    _epoch = 600
    print(
        'epoch:%03d,      recall_@10:%.5f,     recall_@20:%.5f,     ndcg_@10:%.5f,    ndcg_@20:%.5f,   '
        'sedp_@10:%.5f,     sedp_@20:%.5f,    nov_@10:%.5f,    nov_@20:%.5f\n' % (_epoch,
                                                                                  test_recall_bgcf[1],
                                                                                  test_recall_bgcf[2],
                                                                                  test_ndcg_bgcf[1],
                                                                                  test_ndcg_bgcf[2],
                                                                                  test_sedp[0],
                                                                                  test_sedp[1],
                                                                                  test_nov[1],
                                                                                  test_nov[2]))

if __name__ == '__main__':

    if args.infer == 'eval':
        eval_sdk()
    elif args.infer == 'infer':
        run()
