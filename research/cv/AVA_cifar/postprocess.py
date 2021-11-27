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
from src.config import get_config
from src.knn_eval import KnnEval

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--result_dir", type=str, default="./result_Files",
                    help="infer result dataset directory")
parser.add_argument("--label_dir", type=str, default="",
                    help="label data file")
parser.add_argument("--training_dir", type=str, default="",
                    help="training data file")
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    bs = config.batch_size
    knn_acc = KnnEval(batch_size=bs, device_num=1)
    result_num = len(os.listdir(args_opt.result_dir))
    label_list = np.load(args_opt.label_dir)
    training_list = np.load(args_opt.training_dir)
    for i in range(result_num):
        f = "ava_bs" + str(bs) + "_" + str(i) + "_0.bin"
        feature = np.fromfile(os.path.join(args_opt.result_dir, f), np.float32)
        feature = Tensor(feature.reshape(bs, config.low_dims))
        label = Tensor(label_list[i])
        training = Tensor(training_list[i])
        inputs = (feature, label, training)
        knn_acc.update(*inputs)
    knn_acc = float(knn_acc.eval())
    print("The knn result is {}.".format(knn_acc))
