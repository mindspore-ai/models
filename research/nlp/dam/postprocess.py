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

"""
DAM postprocess script.
"""

import os
import argparse
import numpy as np

from mindspore import Tensor

from src.metric import EvalMetric

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--model_name", type=str, default="DAM_ubuntu", help="The model name.")
parser.add_argument("--batch_size", type=int, default=200, help="Eval batch size, default is 200, 256 for DAM_douban")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")

args = parser.parse_args()


if __name__ == "__main__":
    metric = EvalMetric(model_name=args.model_name)
    metric.clear()

    file_name = os.listdir(args.label_dir)

    for f in file_name:
        rst_file = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        label_file = os.path.join(args.label_dir, f)

        logits = np.fromfile(rst_file, np.float32).reshape(args.batch_size, 1)
        logits = Tensor(logits)

        label_ids = np.fromfile(label_file, np.int32).reshape(args.batch_size, 1)
        label_ids = Tensor(label_ids)

        metric.update(logits, label_ids)
    accuracy = metric.eval()
    print("==============================================================")
    print("accuracy: {}".format(accuracy))
    print("==============================================================")
