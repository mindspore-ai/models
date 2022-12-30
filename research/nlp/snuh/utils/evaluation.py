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

"""Evaluation utils."""
import numpy as np
import mindspore.ops as ops
import mindspore.numpy as mindnp
transpose = ops.Transpose()
matmul = ops.MatMul()
mind_topk = ops.TopK(sorted=True)

def compute_retrieval_precision(train_loader, eval_loader,
                                encode_discrete=None, distance_metric='hamming',
                                num_retrieve=100, num_features=128):
    """compute precision"""
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, labels) in loader:
            encoding_chunks.append(docs if encode_discrete is None else
                                   encode_discrete(docs))
            label_chunks.append(labels)

        encoding_mat = mindnp.concatenate(encoding_chunks, axis=0)
        label_mat = mindnp.concatenate(label_chunks, axis=0).asnumpy()
        label_lists = [[j for j in np.nonzero(label_mat[i])[0]] for i in
                       range(label_mat.shape[0])]
        return encoding_mat, label_lists

    src_encodings, src_label_lists = extract_data(train_loader)
    tgt_encodings, tgt_label_lists = extract_data(eval_loader)

    prec = compute_topk_average_precision(tgt_encodings, tgt_label_lists,
                                          src_encodings, src_label_lists,
                                          num_retrieve, distance_metric, num_features)
    return prec

def compute_topk_average_precision(tgt_encodings, tgt_label_lists,
                                   src_encodings, src_label_lists,
                                   num_retrieve, distance_metric='hamming',
                                   num_features=128, chunk_size=100):
    """compute average precision"""
    k = min(num_retrieve, len(src_encodings))
    d = compute_distance(tgt_encodings, src_encodings, distance_metric,
                         chunk_size)
    _, list_topk_nearest_indices = mind_topk(num_features-d, k)

    average_precision = 0.
    for i, topk_nearest_indices in enumerate(list_topk_nearest_indices.asnumpy()):
        gold_set = set(tgt_label_lists[i])
        candidate_lists = [src_label_lists[j] for j in topk_nearest_indices]
        precision = len([_ for candidates in candidate_lists
                         if not gold_set.isdisjoint(candidates)]) / k * 100
        average_precision += precision / tgt_encodings.shape[0]
    return average_precision

def compute_distance(x1, x2, distance_metric='hamming', chunk_size=1000):
    if distance_metric == 'hamming':
        d = compute_hamming_distance(x1, x2, chunk_size=chunk_size)
    else:
        raise Exception('Unsupported distance: {0}'.format(distance_metric))
    return d

def compute_hamming_distance(x1, x2, chunk_size=100):
    """compute hamming distance"""
    assert x1.shape[1] == x2.shape[1]

    d = []
    for i in range(0, x1.shape[0], chunk_size):
        x1_chunk = x1[i:i + chunk_size]

        a = matmul((1 - x1_chunk), transpose(x2, (1, 0)))  # X2 one, X1_chunk zero
        b = matmul(x1_chunk, transpose((1 - x2), (1, 0)))  # X1_chunk one, X2 zero
        d.append(a + b)

    return mindnp.concatenate(d, axis=0)  # N x M
