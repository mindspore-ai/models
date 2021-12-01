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
sampler for length bucketing (batch by tokens)
"""
import random
import json
import gc

from cytoolz import partition_all


class TokenBucketSampler:
    """
        Data sampler for token bucket
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i + self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class TokenBucketPathSampler:
    """
        Sampler for token bucket path
    """
    def __init__(self, path_lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8, use_data_fix=False):
        self.path_lens = path_lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple
        self.use_data_fix = use_data_fix

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        with open(self.path_lens) as f:
            self._lens = json.load(f)
        if self.use_data_fix:
            self._lens = {key: 30 for key in self._lens}
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i + self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        del self._lens
        gc.collect()
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class TokenSampler:
    """
        Sampler for tokens
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8, use_data_fix=False):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple
        self.use_data_fix = use_data_fix

    def _create_ids(self):
        return list(range(self._lens))

    def _sort_fn(self, i):
        return i

    def __iter__(self):

        ids = self._create_ids()
        # random.shuffle(ids)
        buckets = [sorted(ids[i:i + self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        gc.collect()
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class BatchSampler:
    """
        Batch Sampler
    """
    def __init__(self, lens, batch_size):
        self._lens = lens
        self._batch_size = batch_size

    def _create_ids(self):
        return list(range(self._lens))

    def __iter__(self):
        ids = self._create_ids()
        batches = [ids[i:i + self._batch_size] for i in range(0, len(ids), self._batch_size)]
        gc.collect()
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")
