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
import scipy.io

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../data/BSR/BSDS500/data/images/test",
                        help="path of input images directory")
    parser.add_argument("--pipeline_path", type=str, default="../data/pipeline/hed.pipeline",
                        help="path of pipeline file")
    parser.add_argument("--output_dir", type=str, default="./result/hed_result",
                        help="path of output images directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    hed_infer = SDKInferWrapper()
    hed_infer.load_pipeline(args.pipeline_path)
    path_list = os.listdir(args.input_dir)
    path_list.sort()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_path in path_list:
        print(img_path)
        if img_path.endswith('.jpg') or img_path.endswith('.png') or img_path.endswith('.bin'):
            # 先推理
            result_png, result_mat = hed_infer.do_infer(os.path.join(args.input_dir, img_path))
            filename = img_path.split('/')[-1]
            filename = filename.split('.')[0]
            print(filename)
            try:
                result_path_png = os.path.join(args.output_dir, "{}.png".format(filename))
                print("result_path_png: ", result_path_png)
                result_png.save(result_path_png, quality=95)
            except OSError:
                pass

            try:
                result_path_mat = os.path.join(args.output_dir, "{}.mat".format(filename))
                print("result_path_mat: ", result_path_mat)
                scipy.io.savemat(result_path_mat, {'result': result_mat})
            except OSError:
                pass
