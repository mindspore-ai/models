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
Process datasets

Both the 'SZ-taxi' and 'Los-loop' datasets can be downloaded from the link below:
https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data
"""
import os
import numpy as np
import pandas as pd
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size


class TGCNDataset:
    """
    Custom T-GCN datasets
    """

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


def load_adj_matrix(dataset, abs_path=None, dtype=np.float32):
    """
    Load adjacency matrix from corresponding csv file

    Args:
        dataset(str): name of dataset (the same as folder name)
        abs_path(str): absolute data directory path
        dtype(type): data type (Default: np.float32)

    Returns:
        adj: adjacency matrix in ndarray
    """
    if abs_path is not None:
        path = os.path.join(abs_path, dataset, 'adj.csv')
    else:
        path = os.path.join('data', dataset, 'adj.csv')
    adj_df = pd.read_csv(path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def load_feat_matrix(dataset, abs_path=None, dtype=np.float32):
    """
    Load feature matrix from corresponding csv file

    Args:
        dataset(str): name of dataset (the same as folder name)
        abs_path(str): absolute data directory path
        dtype(type): data type (Default: np.float32)

    Returns:
        feat: feature matrix in ndarray
        max_val: max value in feature matrix
    """
    if abs_path is not None:
        path = os.path.join(abs_path, dataset, 'feature.csv')
    else:
        path = os.path.join('data', dataset, 'feature.csv')
    feat_df = pd.read_csv(path)
    feat = np.array(feat_df, dtype=dtype)
    max_val = np.max(feat)
    return feat, max_val


def generate_dataset_np(feat, seq_len, pre_len, split_ratio, normalize=True, time_len=None):
    """
    Generate ndarrays from matrixes

    Args:
        feat(ndarray): feature matrix
        seq_len(int): length of the train data sequence
        pre_len(int): length of the prediction data sequence
        split_ratio(float): proportion of the training set
        normalize(bool): scale the data to (0, 1], divide by the maximum value in the data
        time_len(int): length of the time series in total

    Returns:
        Train set (inputs, targets) and evaluation set (inputs, targets) in ndarrays
    """
    if time_len is None:
        time_len = feat.shape[0]
    if normalize:
        max_val = np.max(feat)
        feat = feat / max_val
    train_size = int(time_len * split_ratio)
    train_data = feat[0:train_size]
    eval_data = feat[train_size:time_len]
    train_inputs, train_targets, eval_inputs, eval_targets = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_inputs.append(np.array(train_data[i: i + seq_len]))
        train_targets.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(eval_data) - seq_len - pre_len):
        eval_inputs.append(np.array(eval_data[i: i + seq_len]))
        eval_targets.append(np.array(eval_data[i + seq_len: i + seq_len + pre_len]))
    return np.array(train_inputs), np.array(train_targets), np.array(eval_inputs), np.array(eval_targets)


def generate_dataset_ms(config, training, abs_path=None):
    """
    Generate MindSpore dataset from ndarrays

    Args:
        config(ConfigTGCN): configuration of parameters
        training(bool): generate training dataset or evaluation dataset
        abs_path(str): absolute data directory path

    Returns:
        dataset: MindSpore dataset for training/evaluation
    """
    dataset = config.dataset
    seq_len = config.seq_len
    pre_len = config.pre_len
    split_ratio = config.train_split_rate
    batch_size = config.batch_size

    feat, _ = load_feat_matrix(dataset, abs_path)
    train_inputs, train_targets, eval_inputs, eval_targets = generate_dataset_np(feat, seq_len, pre_len, split_ratio)

    if training:
        dataset_generator = TGCNDataset(train_inputs, train_targets)
    else:
        dataset_generator = TGCNDataset(eval_inputs, eval_targets)

    dataset = ds.GeneratorDataset(dataset_generator, ["inputs", "targets"], shuffle=False)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def generate_dataset_ms_distributed(config, training, abs_path=None):
    """
    Generate MindSpore dataset from ndarrays in distributed training

    Args:
        config(ConfigTGCN): configuration of parameters
        training(bool): generate training dataset or evaluation dataset
        abs_path(str): absolute data directory path

    Returns:
        dataset: MindSpore dataset for training/evaluation (distributed)
    """
    dataset = config.dataset
    seq_len = config.seq_len
    pre_len = config.pre_len
    split_ratio = config.train_split_rate
    if training:
        batch_size = config.batch_size
    else:
        batch_size = 1

    # Get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()

    feat, _ = load_feat_matrix(dataset, abs_path)
    train_inputs, train_targets, eval_inputs, eval_targets = generate_dataset_np(feat, seq_len, pre_len, split_ratio)

    if training:
        dataset_generator = TGCNDataset(train_inputs, train_targets)
    else:
        dataset_generator = TGCNDataset(eval_inputs, eval_targets)

    dataset = ds.GeneratorDataset(dataset_generator, ["inputs", "targets"], shuffle=False,
                                  num_shards=rank_size, shard_id=rank_id)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
