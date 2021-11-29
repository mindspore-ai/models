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

"""
preprocess.
"""
import os
import argparse

import numpy as np

from src.writer import writer_data
from src.dataset import get_adj_features_labels

def generate_txt():
    """Generate txt files."""
    def w2txt(file, data):
        s = ""
        for i in range(len(data)):
            s = s + str(data[i])
            s = s + " "
        with open(file, "w") as f:
            f.write(s)

    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--raw_data', type=str, default='./data', help='Raw data directory')
    parser.add_argument('--data_dir', type=str, default='./results/data_mr', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--result_path', type=str, default='./results/data', help='Result path')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    args_opt = parser.parse_args()

    # 训练数据预处理
    if not os.path.exists(os.path.join(args_opt.output_dir, "data_mr")):
        os.makedirs(os.path.join(args_opt.output_dir, "data_mr"))
    writer_data(mindrecord_script=args_opt.dataset,
                mindrecord_file=os.path.join(args_opt.output_dir, "data_mr"),
                mindrecord_partitions=1,
                mindrecord_header_size_by_bit=18,
                mindrecord_page_size_by_bit=20,
                graph_api_args=args_opt.raw_data)

    if not os.path.exists(os.path.join(args_opt.result_path, args_opt.dataset)):
        os.makedirs(os.path.join(args_opt.result_path, args_opt.dataset))

    adj, feature, label_onehot, _ = get_adj_features_labels(os.path.join(args_opt.data_dir, args_opt.dataset))
    adj = (adj.reshape(-1)).astype(np.float32)
    feature = (feature.reshape(-1)).astype(np.float32)
    label_onehot = (label_onehot.reshape(-1)).astype(np.int32)

    w2txt(os.path.join(args_opt.result_path, args_opt.dataset, "adjacency.txt"), adj)
    w2txt(os.path.join(args_opt.result_path, args_opt.dataset, "feature.txt"), feature)
    w2txt(os.path.join(args_opt.result_path, args_opt.dataset, "label_onehot.txt"), label_onehot)

if __name__ == '__main__':
    generate_txt()
