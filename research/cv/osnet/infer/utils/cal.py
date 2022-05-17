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

"""Postprocess script"""
import os
import sys
import argparse
import numpy as np
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
config_path = CURRENT_DIR.rsplit('/', 2)[0]
sys.path.append(config_path)
from model_utils.metrics import distance
from model_utils.metrics import rank
from mindspore import Tensor

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
        mode = file.split("_")[0]
        full_file_path = os.path.join(result_path, file)
        label_file = os.path.join(label_path, file.split(".txt")[0]+".bin")
        pids = np.fromfile(label_file, dtype=np.int32)
        camlabel_file = os.path.join(camlabel_path, file.split(".txt")[0]+".bin")
        camids = np.fromfile(camlabel_file, dtype=np.int64)
        if mode == 'query':
            q_pids.extend(pids)
            q_camids.extend(camids)
            if os.path.isfile(full_file_path):
                result = []
                f1 = open(full_file_path)
                line1 = f1.readline()
                line1 = line1[:-1]
                while line1:
                    line1 = f1.readline()
                    line1 = line1[:-1]
                    if line1[:-1] == '':
                        result.append(0)
                    else:
                        result.append(float(line1[:-1]))
                qf.append(result)
        else:
            g_pids.extend(pids)
            g_camids.extend(camids)
            if os.path.isfile(full_file_path):
                result = []
                f1 = open(full_file_path)
                line1 = f1.readline()
                line1 = line1[:-1]
                while line1:
                    line1 = f1.readline()
                    line1 = line1[:-1]
                    if line1[:-1] == '':
                        result.append(0)
                    else:
                        result.append(float(line1[:-1]))
                gf.append(result)
    qf = Tensor(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting features from query set ...')
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))
    print(qf)
    gf = Tensor(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting features from gallery set ...')
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))
    print(gf)
    print('Computing distance matrix ...')
    distmat = distance.compute_distance_matrix(qf, gf, 'euclidean')
    distmat = distmat.asnumpy()
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

if __name__ == "__main__":
    cal_acc(args.dataset, args.result_path, args.label_file, args.camlabel_file)
