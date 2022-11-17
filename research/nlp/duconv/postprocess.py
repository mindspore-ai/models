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

'''
postprocess script.
'''

import os
import argparse
import numpy as np
from mindspore import Tensor
from src.eval_310 import calc_f1

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_files", help="infer result Files")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    file_name = os.listdir(args.label_dir)
    sents = []
    for f in file_name:
        f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        logits = np.fromfile(f_name, np.float32)
        logits = Tensor(logits).asnumpy()
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, 1)).asnumpy().reshape(-1)
        sents.append([str(logits), str(label_ids)])
    f1 = calc_f1(sents)
    output_str = "F1: %.2f%%\n" % (f1 * 100)
    print("output_str:", output_str)
