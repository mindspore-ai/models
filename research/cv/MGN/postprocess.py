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
"""postprocess for 310 inference"""
import os
import argparse
import numpy as np
from metric_utils.re_ranking import re_ranking
from metric_utils.functions import cmc, mean_ap
from model_utils.config import get_config
config_path = "../configs/market1501_config.yml"
config = get_config()
parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="result files path.")
parser.add_argument("--preprocess_result_dir", type=str, default="./preprocess_Result", help="result files path.")
parser.add_argument("--config_path", type=str, default="../configs/market1501_config.yml", help="config file path.")
args = parser.parse_args()


def txt2list(filename):
    txt_tables = []
    f_read = open(filename, "r", encoding='utf-8')
    line = f_read.readline()
    while line:
        line = int(line)
        txt_tables.append(line)
        line = f_read.readline()
    return txt_tables


if __name__ == '__main__':
    output_path = args.preprocess_result_dir
    output_test_path = os.path.join(output_path, "test")
    output_query_path = os.path.join(output_path, "query")
    test_img_path = args.result_dir
    test_label_path = os.path.join(output_test_path, "market1501_label_ids.npy")
    query_label_path = os.path.join(output_query_path, "market1501_label_ids.npy")

    features = []
    labels = np.load(test_label_path, allow_pickle=True)
    for idx, label in enumerate(labels):

        ff = np.zeros((config.per_batch_size, 2048))

        file_name = "market1501_test_bs" + str(config.per_batch_size) + "_" + str(idx) + "_0_0" + ".bin"
        f_name = os.path.join(test_img_path, file_name)
        f = np.fromfile(f_name, np.float32)
        f = f.reshape(config.per_batch_size, 2048)

        ff = ff + f

        file_name = "market1501_test_bs" + str(config.per_batch_size) + "_" + str(idx) + "_1_0" + ".bin"
        f_name = os.path.join(test_img_path, file_name)
        f = np.fromfile(f_name, np.float32)
        f = f.reshape(config.per_batch_size, 2048)

        ff = ff + f

        fnorm = np.sum(np.sqrt(np.square(ff)), axis=1, keepdims=True)
        ff = ff / fnorm

        features.append(ff)
    gf = np.concatenate(features, axis=0)

    features = []
    labels = np.load(query_label_path, allow_pickle=True)
    for idx, label in enumerate(labels):

        ff = np.zeros((config.per_batch_size, 2048))

        file_name = "market1501_query_bs" + str(config.per_batch_size) + "_" + str(idx) + "_0_0" + ".bin"
        f_name = os.path.join(test_img_path, file_name)
        f = np.fromfile(f_name, np.float32)
        f = f.reshape(config.per_batch_size, 2048)

        ff = ff + f

        file_name = "market1501_query_bs" + str(config.per_batch_size) + "_" + str(idx) + "_1_0" + ".bin"
        f_name = os.path.join(test_img_path, file_name)
        f = np.fromfile(f_name, np.float32)
        f = f.reshape(config.per_batch_size, 2048)

        ff = ff + f

        fnorm = np.sum(np.sqrt(np.square(ff)), axis=1, keepdims=True)
        ff = ff / fnorm

        features.append(ff)
    qf = np.concatenate(features, axis=0)

    file_path = os.path.join(output_test_path, 't_cams.txt')
    t_cams = txt2list(file_path)

    file_path = os.path.join(output_test_path, 't_ids.txt')
    t_ids = txt2list(file_path)

    file_path = os.path.join(output_query_path, 'q_cams.txt')
    q_cams = txt2list(file_path)

    file_path = os.path.join(output_query_path, 'q_ids.txt')
    q_ids = txt2list(file_path)

    re_rank = True
    if re_rank:
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    else:
        dist = cdist(qf, gf)
    r = cmc(dist, q_ids, t_ids, q_cams, t_cams,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)
    m_ap = mean_ap(dist, q_ids, t_ids, q_cams, t_cams)

    print(
        '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
        )
    )
