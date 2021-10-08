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
"""post process for 310 inference"""
import os
import time
import argparse
import numpy as np
from src.re_ranking import re_ranking

parser = argparse.ArgumentParser(description="vehiclenet inference")
parser.add_argument("--test_label", type=str, required=True, help="test label.")
parser.add_argument("--query_label", type=str, required=True, help="query label.")
parser.add_argument("--test_out_path", type=str, required=True, help="test output file path.")
parser.add_argument("--query_out_path", type=str, required=True, help="query output file path.")
args_opt = parser.parse_args()

def prepare(out_path):
    """prepare
    """
    out_files = os.listdir(out_path)
    out_files.sort()

    num = 0
    for file in out_files:
        num = num + 1

    shape = (int(num / 2), 1, 512)
    dataset = np.zeros(shape, np.float32)

    i = 0
    count = 0
    for file in out_files:
        file_name = os.path.join(out_path, file)
        f = open(file_name, mode='rb')
        count = count + 1
        img = np.fromfile(f, dtype=np.float32).reshape(1, 512)
        if count % 2 == 0:
            dataset[i, :] += img
            i = i + 1
        else:
            dataset[i, :] = img

    return dataset

def get_labels(path):
    """get_labels
    """
    label_path = os.path.join(path, 'label')
    camera_path = os.path.join(path, 'camera')

    label_files = os.listdir(label_path)
    label_files.sort()

    num = 0
    for file in label_files:
        num = num + 1

    labels = np.zeros((num), np.int32)

    i = 0
    for file in label_files:
        file_name = os.path.join(label_path, file)
        f = np.load(file_name)
        labels[i] = f[0]
        i = i + 1

    camera_files = os.listdir(label_path)
    camera_files.sort()

    num = 0
    for file in camera_files:
        num = num + 1

    cameras = np.zeros((num), np.int32)

    i = 0
    for file in camera_files:
        file_name = os.path.join(camera_path, file)
        f = np.load(file_name)
        cameras[i] = f[0]
        i = i + 1

    return labels, cameras

def extract_feature(dataset):
    """extract_feature
    """
    shape = dataset.shape
    image_size = shape[0]
    features = np.zeros((image_size, 512), dtype=np.float32)

    idx = 0
    for ff in dataset:
        fnorm = np.linalg.norm(ff, ord=2, axis=1)
        fnorm = fnorm.reshape(ff.shape[0], 1)
        fnorm = np.tile(fnorm, (ff.shape[0], ff.shape[1]))
        ff = np.divide(ff, fnorm)
        features[idx] = ff
        idx = idx + 1

    return features

def calculate_result_rerank(test_feature_, test_label_, test_camera_, query_feature_, query_label_, query_camera_, k1=100, k2=15, lambda_value=0):
    """calculate_result_rerank
    """
    CMC = np.zeros(test_label_.shape, dtype=float)
    AP = 0.0

    since = time.time()
    q_t_dist = np.matmul(query_feature_, test_feature_.transpose((1, 0)))
    q_q_dist = np.matmul(query_feature_, query_feature_.transpose((1, 0)))
    t_t_dist = np.matmul(test_feature_, test_feature_.transpose((1, 0)))

    re_rank = re_ranking(q_t_dist, q_q_dist, t_t_dist, k1, k2, lambda_value)
    time_elapsed = time.time() - since
    print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    for i in range(len(query_label_)):
        AP_tmp, CMC_tmp = evaluate(re_rank[i, :], query_label_[i], query_camera_[i], test_label_, test_camera_)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        AP += AP_tmp

    CMC = CMC / query_label_.shape[0]

    str_result = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f\n' % (CMC[0], CMC[4], CMC[9], AP / query_label_.shape[0])
    print(str_result)

def evaluate(score, query_label_, query_camera_, test_label_, test_camera_):
    """evaluate
    """
    index = np.argsort(score)

    query_index = np.argwhere(test_label_ == query_label_)
    camera_index = np.argwhere(test_camera_ == query_camera_)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(test_label_ == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    """compute_mAP
    """
    ap = 0
    cmc = np.zeros((len(index)), dtype=int)  # 11579

    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)  # different is True, same is False
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)  # different is False, same is True
    rows_good = np.argwhere(np.equal(mask, True))
    rows_good = rows_good.flatten()

    for i in range(len(cmc)):
        if i >= rows_good[0]:
            cmc[i] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

if __name__ == '__main__':
    test_dataset = prepare(args_opt.test_out_path)
    query_dataset = prepare(args_opt.query_out_path)

    test_label, test_camera = get_labels(args_opt.test_label)
    query_label, query_camera = get_labels(args_opt.query_label)

    test_feature = extract_feature(test_dataset)
    query_feature = extract_feature(query_dataset)

    calculate_result_rerank(test_feature, test_label, test_camera, query_feature, query_label, query_camera)
