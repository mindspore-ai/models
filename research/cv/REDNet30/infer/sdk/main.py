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
"""sdk infer"""
import argparse
import os
from infer_wrapper import SDKInferWrapper

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_noise_dir", type=str, default="../data/input_noise_train",
                        help="path of input images directory")
    parser.add_argument("--pipeline_path", type=str, default="./pipeline/red30.pipeline",
                        help="path of pipeline file")
    parser.add_argument("--output_dir", type=str, default="../data/result",
                        help="path of output images directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    red_infer = SDKInferWrapper()
    red_infer.load_pipeline(args.pipeline_path)
    path_list = os.listdir(args.input_noise_dir)
    path_list.sort()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_path in path_list:
        print(img_path)
        result_png = red_infer.do_infer(os.path.join(args.input_noise_dir, img_path))
        filename = img_path.split('/')[-1]
        filename = filename.split('.')[0]
        print(filename)
        try:
            result_path_png = os.path.join(args.output_dir, "{}.jpg".format(filename))
            print("result_path_png: ", result_path_png)
            result_png.save(result_path_png, quality=95)
        except OSError:
            pass
