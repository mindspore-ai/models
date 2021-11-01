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
from src.data_process import load_graph_data
from src.utilities import diffusion_fun_improved_ppmi_dynamic_sparsity, diffusion_fun_sparse
from src.config import ConfigDGCN


def generate_bin():
    """Generate bin files."""
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--data_dir', type=str, default='./data/cora/cora_mr', help='Dataset directory')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='Result path')
    args_opt = parser.parse_args()

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_graph_data(args_opt.data_dir)
    print(shape(y_train), shape(y_val), shape(y_test), shape(train_mask), shape(val_mask))
    adj_path = os.path.join(args_opt.result_path, "00_data")
    ppmi_path = os.path.join(args_opt.result_path, "01_data")
    feature_path = os.path.join(args_opt.result_path, "02_data")
    os.makedirs(adj_path)
    os.makedirs(feature_path)
    os.makedirs(ppmi_path)
    config = ConfigDGCN()
    diffusions = diffusion_fun_sparse(adj.tocsc())
    diffusions = diffusions.toarray()
    ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(adj, path_len=config.path_len, k=1.0)
    ppmi = ppmi.toarray()
    features = features.toarray()
    diffusions = diffusions.astype(np.float16)
    ppmi = ppmi.astype(np.float16)
    features = features.astype(np.float16)

    diffusions.tofile(os.path.join(adj_path, "diffusions.bin"))
    ppmi.tofile(os.path.join(ppmi_path, "ppmi.bin"))
    features.tofile(os.path.join(feature_path, "feature.bin"))
    np.save(os.path.join(args_opt.result_path, 'label_onehot.npy'), labels)
    np.save(os.path.join(args_opt.result_path, 'test_mask.npy'), test_mask)

if __name__ == '__main__':
    generate_bin()
    