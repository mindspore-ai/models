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
"""preprocess"""
import os
import _pickle as cPickle
import argparse
import numpy as np

def generate_bin():
    """Generate bin files."""
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--data_dir', type=str,
                        default='./datasets', help='Dataset directory')
    parser.add_argument('--result_path', type=str,
                        default='./preprocess_Result/', help='Result path')
    args_opt = parser.parse_args()

    feature_bin_path = os.path.join(args_opt.result_path, "00_data")
    label_bin_path = os.path.join(args_opt.result_path, "label")

    if not os.path.exists(feature_bin_path):
        os.makedirs(feature_bin_path)

    if not os.path.exists(label_bin_path):
        os.makedirs(label_bin_path)

    test_path = os.path.join(args_opt.data_dir, "feature_test.pkl")
    feature_bin_path = os.path.join(feature_bin_path, "input.bin")
    label_bin_path = os.path.join(label_bin_path, "label.bin")

    test_file = open(test_path, "rb")
    test_data = cPickle.load(test_file, encoding="bytes")


    feature = np.array(test_data[0], dtype=np.float32)
    feature.tofile(feature_bin_path)

    label = np.array(test_data[1], dtype=np.float32)
    label.tofile(label_bin_path)


if __name__ == '__main__':
    generate_bin()
