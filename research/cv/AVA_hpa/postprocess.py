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
"""postprocess."""
import os
import argparse
import numpy as np

from mindspore import Tensor
from src.network_define_eval import EvalMetric

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--result_dir", type=str, default="./result_Files",
                    help="infer result dataset directory")
parser.add_argument("--label_dir", type=str, default="",
                    help="label data file")
parser.add_argument("--nslice_dir", type=str, default="",
                    help="nslice data file")
parser.add_argument("--save_eval_path", type=str, default="./eval_result",
                    help="eval result path")
parser.add_argument("--classes", type=int, default=10, help='class number')
args_opt = parser.parse_args()

if __name__ == '__main__':
    batch_size = 1
    bag_size_for_eval = 20
    acc = EvalMetric(path=args_opt.save_eval_path)
    result_num = len(os.listdir(args_opt.result_dir))
    label_list = np.load(args_opt.label_dir)
    nslice_list = np.load(args_opt.nslice_dir)
    for i in range(result_num):
        f = "ava_bs" + str(batch_size) + "_" + str(i) + "_0.bin"
        feature = np.fromfile(os.path.join(args_opt.result_dir, f), np.float32)
        feature = Tensor(feature.reshape(bag_size_for_eval, args_opt.classes))
        label = Tensor(label_list[i])
        nslice = Tensor(nslice_list[i])
        inputs = (feature, label, nslice)
        acc.update(*inputs)
    result_return = acc.eval()
    print("The result is {}.".format(result_return))
