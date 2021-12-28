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
""" DataLoader """
import os
import numpy as np
import mindspore.dataset as ds


class TrainDataset:
    """ Get Training data """

    def __init__(self, triples, num_entity, num_relation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = np.sqrt(1 / np.array([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_entity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = np.array(negative_sample, dtype=np.int32)

        positive_sample = np.array(positive_sample, dtype=np.int32)

        subsampling_weight = np.array(subsampling_weight, dtype=np.float32)

        return positive_sample, negative_sample, subsampling_weight

    @staticmethod
    def count_frequency(triples, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset:
    """ Get Training data """

    def __init__(self, triples, all_true_triples, num_entity, num_relation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.num_entity = num_entity
        self.num_relation = num_relation
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
        filter_bias = tmp[:, 0]
        negative_sample = tmp[:, 1]

        positive_sample = np.array((head, relation, tail))

        return positive_sample, negative_sample, filter_bias


class BidirectionalOneShotIterator:
    """
    Bidirectional dataloader generates different negative sample data batch-by-batch.

    Args:
        dataloader_head (GeneratorDataset): head entity negative sample dataloader
        dataloader_tail (GeneratorDataset): tail entity negative sample dataloader

    Returns:
        iterator of different negative sample data.

    """
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            positive_sample, negative_sample, subsampling_weight = next(self.iterator_head)
            mode = 'head-mode'
        else:
            positive_sample, negative_sample, subsampling_weight = next(self.iterator_tail)
            mode = 'tail-mode'
        return positive_sample, negative_sample, subsampling_weight, mode

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a GeneratorDataset into python iterator
        """
        while True:
            for data in dataloader.create_dict_iterator():
                yield data["positive"], data["negative"], data["weight"]


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


def create_dataset(data_path, config, is_train, device_num=None, rank_id=None):
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
    num_entity, num_relation = len(entity2id), len(relation2id)
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

    if is_train:
        train_dataloader_head = ds.GeneratorDataset(
            source=TrainDataset(
                triples=train_triple,
                num_entity=len(entity2id),
                num_relation=len(relation2id),
                negative_sample_size=config.negative_sample_size,
                mode="head-batch"
            ),
            shuffle=True,
            num_shards=device_num,
            shard_id=rank_id,
            column_names=["positive", "negative", "weight"]
        ).batch(batch_size=config.batch_size)
        train_dataloader_tail = ds.GeneratorDataset(
            source=TrainDataset(
                triples=train_triple,
                num_entity=len(entity2id),
                num_relation=len(relation2id),
                negative_sample_size=config.negative_sample_size,
                mode="tail-batch"
            ),
            shuffle=True,
            num_shards=device_num,
            shard_id=rank_id,
            column_names=["positive", "negative", "weight"]
        ).batch(batch_size=config.batch_size)
        train_dataloader = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        return num_entity, num_relation, train_dataloader
    triples = test_triple
    test_dataloader_head = ds.GeneratorDataset(
        TestDataset(
            triples=triples,
            all_true_triples=all_true_triples,
            num_entity=num_entity,
            num_relation=num_relation,
            mode='head-batch'
        ),
        column_names=["positive", "negative", "filter_bias"],
        num_parallel_workers=5
    ).batch(config.test_batch_size)
    test_dataloader_tail = ds.GeneratorDataset(
        TestDataset(
            triples=triples,
            all_true_triples=all_true_triples,
            num_entity=num_entity,
            num_relation=num_relation,
            mode='tail-batch'
        ),
        column_names=["positive", "negative", "filter_bias"],
        num_parallel_workers=5
    ).batch(config.test_batch_size)
    return num_entity, num_relation, test_dataloader_head, test_dataloader_tail
