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
"""eval_utils.py"""

import time
from collections import OrderedDict
import numpy as np

from src.model_utils.config import config
from src.meters import AverageMeter

def apply_eval(eval_param_dict):
    """apply eval"""
    net = eval_param_dict["net"]
    net.set_train(False)
    query_dataset = eval_param_dict["query_dataset"]
    gallery_dataset = eval_param_dict["gallery_dataset"]
    query_set = eval_param_dict["query_set"]
    gallery_set = eval_param_dict["gallery_set"]
    print('extracting query features\n')
    query_features, _ = extract_features(net, query_dataset)
    print('extracting gallery features\n')
    gallery_features, _ = extract_features(net, gallery_dataset)
    x = [query_features[fid] for _, fid, _, _ in query_set]
    x = list(map(lambda item: np.expand_dims(item, 0), x))
    x = np.concatenate(x, axis=0)
    y = [gallery_features[fid] for _, fid, _, _ in gallery_set]
    y = list(map(lambda item: np.expand_dims(item, 0), y))
    y = np.concatenate(y, axis=0)
    m, n = x.shape[0], y.shape[0]
    query_feature = x.reshape(m, -1)
    gallery_feature = y.reshape(n, -1)
    query_label = [pid for _, _, pid, _ in query_set]
    gallery_label = [pid for _, _, pid, _ in gallery_set]
    query_cam = [cam for _, _, _, cam in query_set]
    gallery_cam = [cam for _, _, _, cam in gallery_set]
    query_label = np.asarray(query_label)
    gallery_label = np.asarray(gallery_label)
    query_cam = np.asarray(query_cam)
    gallery_cam = np.asarray(gallery_cam)
    CMC = np.zeros((len(gallery_label),))
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], \
gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC / len(query_label)
    cmc_scores = {config.dataset_name: CMC}
    mAP = ap / len(query_label)
    return mAP, cmc_scores

def extract_features(model, dataset, print_freq=10):
    """extract query features/ gallery features"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    for i, (imgs, fids, pids, _) in enumerate(dataset):
        data_time.update(time.time() - end)
        _, g_features, h_features = model(imgs)
        features_to_use = None
        if config.use_G_feature:
            features_to_use = g_features
        else: #use H feature
            features_to_use = h_features
        for fid, feature, pid in zip(fids, features_to_use, pids):
            fid = int(fid.asnumpy())
            features[fid] = feature.asnumpy()
            labels[fid] = pid
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, dataset.get_dataset_size(),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels

def evaluate(qf, ql, qc, gf, gl, gc):
    """evaluate"""
    query = qf
    score = None
    index = None
    score = np.dot(gf, query)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    """compute mAP"""
    ap = 0
    cmc = np.zeros((len(index),))
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc
