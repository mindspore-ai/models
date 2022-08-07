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
import numpy as np
from PIL import Image
import mindspore.nn as nn

def parse(arg=None):
    """Define configuration of postprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, default='./result_Files/')
    parser.add_argument('--mask_path', type=str, default='./preprocess_Mask_Result/')
    parser.add_argument('--output_dir', type=str, default='./postprocess_Result/')
    return parser.parse_args(arg)

def load_bin_file(bin_file, shape=None, dtype="float32"):
    """Load data from bin file"""
    data = np.fromfile(bin_file, dtype=dtype)
    if shape:
        data = np.reshape(data, shape)
    return data

def save_bin_to_image(data, out_name):
    """Save bin file to image arrays"""
    pic = Image.fromarray(data)
    pic = pic.convert('RGB')
    pic.save(out_name)
    print("Successfully save image in " + out_name)

def scan_dir(bin_path):
    """Scan directory"""
    out = os.listdir(bin_path)
    return out

def sigmoid(z):
    """sigmoid"""
    return 1/(1 + np.exp(-z))

def postprocess(args):
    """Post process bin file"""
    file_list = scan_dir(args.bin_path)
    loss = nn.Loss()
    F_score = nn.F1()
    loss.clear()
    total_test_step = 0
    test_data_size = len(file_list)
    for file_path in file_list:
        data = load_bin_file(args.bin_path + file_path, shape=(224, 224), dtype="float32")
        targets1 = load_bin_file(args.mask_path + file_path, shape=(224, 224), dtype="float32")
        pre_mask = data
        targets1 = targets1.astype(int)
        pre_mask = pre_mask.flatten()
        targets1 = targets1.flatten()
        pre_mask1 = pre_mask.tolist()
        F_pre = np.array([[1 - i, i] for i in pre_mask1])
        F_score.update(F_pre, targets1)
        total_test_step = total_test_step + 1
        if total_test_step % 100 == 0:
            print("evaling:{}/{}".format(total_test_step, test_data_size))
    F_score_result = F_score.eval()
    print("F-score: ", (F_score_result[0] + F_score_result[1]) / 2)
    print("---------------eval finish------------")
if __name__ == "__main__":
    argms = parse()
    postprocess(argms)
