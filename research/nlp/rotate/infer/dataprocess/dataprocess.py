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
dataprocess
"""

import os
from tqdm import tqdm
import numpy as np
from config import config


class TestDataset:
    """ Get Training data """

    def __init__(self, triples, all_true_triples, num_entity_, num_relation_, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.num_entity = num_entity_
        self.num_relation = num_relation_
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.num_entity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.num_entity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = np.array(tmp)
        filter_bias1 = tmp[:, 0]
        negative_sample1 = tmp[:, 1]

        positive_sample1 = np.array((head, relation, tail))

        return positive_sample1, negative_sample1, filter_bias1

def read_triple(file_path, entity2id, relation2id):
    """
    Read triples and map them into ids.

    Args:
        file_path (str): data file path
        entity2id (dict): entity <--> entity ID
        relation2id (dict): relation <--> relation ID

    Returns:
        triples (list): [(head_id, rel_id, tail_id), ... ]

    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def get_entity_and_relation(data_path):
    """
    Get the map of entity and relation

    Args:
        data_path: data file path

    Returns:
        entity2id (dict): entity <--> entity ID
        relation2id (dict): relation <--> relation ID

    """
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    return entity2id, relation2id

def create_dataset(data_path, config1, is_train, device_num=None, rank_id=None):
    """
    Create GeneratorDataset for train and eval

    Args:
        data_path (str): data file path
        config (class): global hyper-parameters.
        device_num (int): if device_num>1, run parallel; else run on single device.
        rank_id (int): device id
        is_train (bool): train or eval

    Returns:
        ds.GeneratorDataset

    """
    entity2id, relation2id = get_entity_and_relation(data_path)
    num_entity1, num_relation1 = len(entity2id), len(relation2id)
    train_triple = read_triple(
        file_path=os.path.join(data_path, "train.txt"),
        entity2id=entity2id,
        relation2id=relation2id
    )
    valid_triple = read_triple(
        file_path=os.path.join(data_path, "valid.txt"),
        entity2id=entity2id,
        relation2id=relation2id
    )
    test_triple = read_triple(
        file_path=os.path.join(data_path, "test.txt"),
        entity2id=entity2id,
        relation2id=relation2id
    )
    all_true_triples = train_triple + valid_triple + test_triple
    triples = test_triple
    test_dataloader_head_ = TestDataset(triples=triples,
                                        all_true_triples=all_true_triples,
                                        num_entity_=num_entity1,
                                        num_relation_=num_relation1,
                                        mode='head-batch')

    test_dataloader_tail_ = TestDataset(triples=triples,
                                        all_true_triples=all_true_triples,
                                        num_entity_=num_entity1,
                                        num_relation_=num_relation1,
                                        mode='tail-batch')
    return num_entity1, num_relation1, test_dataloader_head_, test_dataloader_tail_

num_entity, num_relation, test_dataloader_head, test_dataloader_tail = create_dataset(
    data_path=config.data_path,
    config1=config,
    is_train=False
)
config.num_entity, config.num_relation = num_entity, num_relation

positive_sample, negative_sample, filter_bias = [], [], []
for i in tqdm(test_dataloader_head):
    positive_sample.append(i[0])
    negative_sample.append(i[1])
    filter_bias.append(i[2])

positive_sample = np.array(positive_sample)
negative_sample = np.array(negative_sample)
filter_bias = np.array(filter_bias)

np.savetxt(config.data_path+'/positive_sample_head.txt', positive_sample, fmt='%i', delimiter=' ')
np.savetxt(config.data_path+'/negative_sample_head.txt', negative_sample, fmt='%i', delimiter=' ')
np.savetxt(config.data_path+'/filter_bias_head.txt', filter_bias, fmt='%i', delimiter=' ')

positive_sample, negative_sample, filter_bias = [], [], []
for i in tqdm(test_dataloader_tail):
    positive_sample.append(i[0])
    negative_sample.append(i[1])
    filter_bias.append(i[2])

positive_sample = np.array(positive_sample)
negative_sample = np.array(negative_sample)
filter_bias = np.array(filter_bias)

np.savetxt(config.data_path+'/positive_sample_tail.txt', positive_sample, fmt='%i', delimiter=' ')
np.savetxt(config.data_path+'/negative_sample_tail.txt', negative_sample, fmt='%i', delimiter=' ')
np.savetxt(config.data_path+'/filter_bias_tail.txt', filter_bias, fmt='%i', delimiter=' ')
