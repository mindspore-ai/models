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

"""Postprocess script"""

import os
import argparse
import numpy as np


from model_utils.metrics import rank

parser = argparse.ArgumentParser(description='postprocess for osnet')
parser.add_argument("--dataset", type=str, default="market1501", help="dataset")
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_file", type=str, required=True, help="label file path")
parser.add_argument("--camlabel_file", type=str, required=True, help="camlabel file path")
args = parser.parse_args()

def cal_acc(dataset, result_path, label_path, camlabel_path):
    '''process features for calculating metrics.'''
    qf, q_pids, q_camids = [], [], []
    gf, g_pids, g_camids = [], [], []

    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(1, -1)
            label_file = os.path.join(label_path, file.split(".bin")[0][:-2] + ".bin")
            pids = np.fromfile(label_file, dtype=np.int32)
            camlabel_file = os.path.join(camlabel_path, file.split(".bin")[0][:-2] + ".bin")
            camids = np.fromfile(camlabel_file, dtype=np.int64)
            mode = file.split("_")[0]
            if mode == "query":
                qf.append(result)
                q_pids.extend(pids)
                q_camids.extend(camids)
            else:
                gf.append(result)
                g_pids.extend(pids)
                g_camids.extend(camids)

    qf = tuple(qf)
    qf = np.concatenate(qf, axis=0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting features from query set ...')
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    gf = tuple(gf)
    gf = np.concatenate(gf, axis=0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting features from gallery set ...')
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    print('Computing distance matrix ...')
    distmat = compute_distance_matrix(qf, gf)

    print('Computing CMC and mAP ...')
    cmc, mAP = rank.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_metric_cuhk03=False
    )

    print('** Results **')
    print('Dataset:{}'.format(dataset))
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    ranks = [1, 5, 10, 20]
    i = 0
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[i]))
        i += 1

def compute_distance_matrix(query_features, gallery_features):
    '''
    compute distance matrix from query_features and gallery_fearutrs
    '''
    m = query_features.shape[0]
    n = gallery_features.shape[0]
    mat1 = np.sum(np.power(query_features, 2), axis=1, keepdims=True)
    mat2 = np.sum(np.power(gallery_features, 2), axis=1, keepdims=True)
    mat1 = np.repeat(mat1, n, axis=1)
    mat2 = np.repeat(mat2, m, axis=1).T
    distmat = mat1 + mat2
    input1 = query_features.astype(np.float16)
    input2 = gallery_features.astype(np.float16)
    output = np.matmul(input1, input2.T)
    output = output.astype(np.float32)
    distmat = distmat - 2 * output
    return distmat

if __name__ == "__main__":
    cal_acc(args.dataset, args.result_path, args.label_file, args.camlabel_file)
