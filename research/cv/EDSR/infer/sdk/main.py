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
"""run sdk infer"""
import argparse
import os
from sr_infer_wrapper import SRInferWrapper

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../data/DIV2K/input/",
                        help="path of input images directory")
    parser.add_argument("--pipeline_path", type=str, default="../data/config/edsr.pipeline",
                        help="path of pipeline file")
    parser.add_argument("--output_dir", type=str, default="../data/sdk_out/",
                        help="path of output images directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    sr_infer = SRInferWrapper()
    sr_infer.load_pipeline(args.pipeline_path)
    path_list = os.listdir(args.input_dir)
    path_list.sort()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_path in path_list:
        print(img_path)
        res = sr_infer.do_infer(os.path.join(args.input_dir, img_path))
        res.save(os.path.join(args.output_dir, img_path.replace('x2', '_infer')))
