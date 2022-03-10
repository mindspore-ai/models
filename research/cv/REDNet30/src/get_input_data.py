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
"""get input data."""
import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/BSD200', help='evaling image path')
    parser.add_argument('--output_path', type=str, default='./data/BSD200_jpeg_quality10', help='output image path')
    opt = parser.parse_args()
    images_dir = "./data/BSD200"
    path = "./data/BSD200_jpeg_quality10"

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    # data
    files = glob.glob(opt.dataset_path + '/*')

    for file in tqdm(files):
        name = file.split("/")[-1]
        img = Image.open(file)
        img.save(os.path.join(opt.output_path, name), format='jpeg', quality=10)
    print("finished!")
