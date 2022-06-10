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

"""Evaluation for NIMA"""
import argparse
import scipy.stats
import numpy as np

def data_post_processing(val_label_path, result_path):
    """ evaluate the result """
    dic = {}
    with open(val_label_path) as l:
        for lst in l.readlines():
            dic[lst.split(',')[1]] = float(lst.split(',')[-1])

    with open(result_path) as f:
        y_pred = f.readlines()
    scores = []
    gt = []
    SCORE_LIST = np.array([x for x in range(1, 11)])

    for i in y_pred:
        pic = i.split(':')[0]
        score_list = [float(j) for j in i.split(':')[1].split()[0:]]
        score = np.sum(np.array(score_list) * SCORE_LIST)
        scores.append(score)
        gt.append(dic[pic])
    scores = np.array(scores)
    gt = np.array(gt)
    result = sum([(scores > 5) & (gt > 5)][0]) + sum([(scores <= 5) & (gt <= 5)][0])
    print('mse:', np.mean(np.power((scores - gt), 2)))
    print('acc: ', result/gt.shape[0])
    print('SRCC: ', scipy.stats.spearmanr(scores, gt)[0])


def run():
    """ run """
    parser = argparse.ArgumentParser(description="input parameter.")
    parser.add_argument("--val_label_path", type=str,
                        help="the path of val label file")
    parser.add_argument("--result_file", type=str,
                        help="the path of infer result.")
    args = parser.parse_args()
    data_post_processing(args.val_label_path, args.result_file)


if __name__ == '__main__':
    run()
