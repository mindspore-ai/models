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
from PIL import Image

def parse(arg=None):
    """Define configuration of postprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, default='./result_Files/')
    parser.add_argument('--target_path', type=str, default='./result_Files/')
    return parser.parse_args(arg)

def load_bin_file(bin_file, dtype="float32"):
    """Load data from bin file"""
    data = np.fromfile(bin_file, dtype=dtype)
    return data

def decode_image(img):
    """Decode a Tensor to image numpy array."""
    if img.shape[0] == 1:
        img = (img[0] + 1) / 2.0 * 255.0
    elif img.shape[0] == 3:
        img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img

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

def postprocess(bin_path):
    """Post process bin file"""
    file_list = scan_dir(bin_path)
    print(file_list)
    for file_path in file_list:
        data = load_bin_file(bin_path + file_path, dtype="float32")
        data = data.reshape((1, 512, 512))
        pos = file_path.find(".")
        file_name = file_path[0:pos] + "." + "jpg"
        outfile = os.path.join(args.target_path, file_name)
        data = decode_image(data)
        save_bin_to_image(data, outfile)

if __name__ == "__main__":
    args = parse()
    postprocess(args.bin_path)
