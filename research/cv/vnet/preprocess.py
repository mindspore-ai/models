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
"""pre process for 310 inference"""
import os
import argparse
from src.config import vnet_cfg as cfg
from src.dataset import InferImagelist
parser = argparse.ArgumentParser('preprocess')
parser.add_argument("--data_path", type=str, default="./promise", help="Path of dataset, default is ./promise")
parser.add_argument("--split_file_path", type=str, default="./split/eval.csv",
                    help="Path of dataset, default is ./split/eval.csv")
args = parser.parse_args()


if __name__ == '__main__':
    result_path = "./results/bin_data"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    dataInferlist = InferImagelist(cfg, args.data_path, args.split_file_path)
    dataManagerInfer = dataInferlist.dataManagerInfer
    for i in range(dataInferlist.__len__()):
        img_data, img_id = dataInferlist.__getitem__(i)
        file_name = img_id + ".bin"
        img_file_path = os.path.join(result_path, file_name)
        img_data.asnumpy().tofile(img_file_path)
    print("=" * 20, "export bin files for 10 test cases finished", "=" * 20)
