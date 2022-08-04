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


"""post process for 310 inference"""
import os
import numpy as np
from src.model_utils.config import config
from src.eval_utils import evaluate

def apply_eval(query_feature, gallery_feature, query_label, gallery_label, query_cam, gallery_cam):
    """Compute CMC and mAP"""
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
    mAP = ap / len(query_label)
    return mAP, CMC

def postprocess():
    """Postprocess"""
    feature_dim = 256
    if config.use_G_feature:
        feature_dim = 2048
    query_pid_path = os.path.join(config.preprocess_result_path, "query", "pid")
    query_camid_path = os.path.join(config.preprocess_result_path, "query", "camid")
    query_prediction_path = config.query_prediction_path
    query_file_names = os.listdir(query_pid_path)
    query_pids = []
    query_camids = []
    query_predictions = []
    for f in query_file_names:
        pid = np.fromfile(os.path.join(query_pid_path, f), dtype=np.int32).tolist()
        camid = np.fromfile(os.path.join(query_camid_path, f), dtype=np.int32).tolist()
        prediction = np.fromfile(os.path.join(query_prediction_path, f), \
                                 dtype=np.float32).reshape((1, feature_dim, 6, 1))
        query_pids += pid
        query_camids += camid
        query_predictions.append(prediction)
    query_pids = np.asarray(query_pids)
    query_camids = np.asarray(query_camids)
    query_predictions = np.concatenate(query_predictions, axis=0)
    query_predictions = query_predictions.reshape(query_predictions.shape[0], -1)
    gallery_pid_path = os.path.join(config.preprocess_result_path, "gallery", "pid")
    gallery_camid_path = os.path.join(config.preprocess_result_path, "gallery", "camid")
    gallery_prediction_path = config.gallery_prediction_path
    gallery_file_names = os.listdir(gallery_pid_path)
    gallery_pids = []
    gallery_camids = []
    gallery_predictions = []
    for f in gallery_file_names:
        pid = np.fromfile(os.path.join(gallery_pid_path, f), dtype=np.int32).tolist()
        camid = np.fromfile(os.path.join(gallery_camid_path, f), dtype=np.int32).tolist()
        prediction = np.fromfile(os.path.join(gallery_prediction_path, f), \
                                 dtype=np.float32).reshape((1, feature_dim, 6, 1))
        gallery_pids += pid
        gallery_camids += camid
        gallery_predictions.append(prediction)
    gallery_pids = np.asarray(gallery_pids)
    gallery_camids = np.asarray(gallery_camids)
    gallery_predictions = np.concatenate(gallery_predictions, axis=0)
    gallery_predictions = gallery_predictions.reshape(gallery_predictions.shape[0], -1)
    mAP_score, CMC_score = apply_eval(query_predictions, gallery_predictions, query_pids, \
                                      gallery_pids, query_camids, gallery_camids)
    # print metrics
    print('Mean AP: {:4.1%}'.format(mAP_score), flush=True)
    cmc_topk = (1, 5, 10)
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, CMC_score[k - 1]), flush=True)

if __name__ == "__main__":
    postprocess()
