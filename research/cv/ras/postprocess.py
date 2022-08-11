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
import argparse
import cv2
import numpy as np

def parse(arg=None):
    """Define configuration of postprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, default='./result_Files/')
    parser.add_argument('--target_path', type=str, default='./result_Files/')
    parser.add_argument('--shape_path', type=str, default='./preprocess_Data/data_shape/')
    return parser.parse_args(arg)

def load_bin_file(bin_file, shape=None, dtype="float32"):
    """Load data from bin file"""
    data = np.fromfile(bin_file, dtype=dtype)
    if shape:
        data = np.reshape(data, shape)
    return data

def scan_dir(bin_path):
    """Scan directory"""
    out = os.listdir(bin_path)
    return out

def sigmoid(z):
    """sigmoid"""
    return 1/(1 + np.exp(-z))

def postprocess():
    """Post process bin file"""
    file_list = scan_dir(args.bin_path)
    for file_path in file_list:
        data = load_bin_file(args.bin_path + file_path, shape=(352, 352), dtype="float32")
        data_shape = load_bin_file(args.shape_path + file_path, dtype="int64")
        img = cv2.resize(data, (int(data_shape[1]), int(data_shape[0])))
        img = sigmoid(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 255
        file_name = file_path.split(".")[0] + ".jpg"
        outfile = os.path.join(args.target_path, file_name)
        cv2.imwrite(outfile, img)
        print("Successfully save image in " + outfile)

if __name__ == "__main__":
    args = parse()
    postprocess()
