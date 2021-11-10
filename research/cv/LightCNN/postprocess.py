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
import argparse
import numpy as np
from eval_lfw import lfw_eval
from eval_blufr import blufr_eval


def extract_features_to_dict_lfw(result_path):
    """get features"""
    files = os.listdir(result_path)
    lfw_files = []
    for file in files:
        if file[:4] == "lfw_" and file[-6:] == "_1.bin":
            lfw_files.append(file)
    features_shape = (len(lfw_files), 256)
    features = np.empty(features_shape, dtype='float32', order='C')

    for idx, file in enumerate(lfw_files):
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            features[idx:idx + 1, :] = np.fromfile(full_file_path, dtype=np.float32)
    feature_dict = {'features': features}
    return feature_dict


def extract_features_to_dict_blufr(result_path):
    """get features"""
    files = os.listdir(result_path)
    blufr_files = []
    for file in files:
        if file[:5] == "blufr" and file[-6:] == "_1.bin":
            blufr_files.append(file)
    features_shape = (len(blufr_files), 256)
    features = np.empty(features_shape, dtype='float32', order='C')

    for idx, file in enumerate(blufr_files):
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            features[idx:idx + 1, :] = np.fromfile(full_file_path, dtype=np.float32)
    feature_dict = {'Descriptors': features}
    return feature_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument("--result_path", type=str, required=True, help="result files path.")
    parser.add_argument("--mat_files_path", type=str, required=True, help="mat files path.")
    args = parser.parse_args()

    dic_lfw = extract_features_to_dict_lfw(args.result_path)
    print("lfw 6,000 pairs test start")
    lfw_eval(dic_lfw, lfw_pairs_mat_path=args.mat_files_path + "/lfw_pairs.mat")
    print("lfw 6,000 pairs test finished")

    dic_blufr = extract_features_to_dict_blufr(args.result_path)
    print("lfw BLUFR protocols test start")
    blufr_eval(dic_blufr, config_file_path=args.mat_files_path + "/blufr_lfw_config.mat")
    print("lfw BLUFR protocols test finished")
