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

import os
import numpy as np
from mindspore.dataset import GeneratorDataset
from src.model.vocabulary import Vocab


class Generator:
    # LM1B dataset
    def __init__(self, _data, batch_size, tgt_len, ext_len=None):
        super(Generator, self).__init__()
        self.bsz = batch_size
        self.bptt = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = _data.size // self.bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        _data = _data[:self.n_step * self.bsz]

        # Evenly divide the data across the bsz batches.
        self._data = _data.reshape(self.bsz, -1).T
        self._data = self._data.astype(np.int32)

        # Number of mini-batches
        self.n_batch = self.n_step // self.bptt

    def __getitem__(self, item):
        item *= self.bptt
        _seq_len = min(self.bptt, self._data.size - 1 - item)

        end_idx = item + _seq_len
        beg_idx = max(0, item - self.ext_len)

        _data = self._data[beg_idx: end_idx]
        _target = self._data[item + 1:item + 1 + _seq_len]
        return _data, _target

    def __len__(self):
        return self.n_batch


class VariableGenerator(Generator):
    def __init__(self, _data, batch_size, tgt_len, ext_len=None, start=0, std=5, min_len=5, max_deviation=3):
        super(VariableGenerator, self).__init__(_data, batch_size, tgt_len, ext_len)
        self.start = start
        self.std = std
        self.min_len = min_len
        self.max_deviation = max_deviation
        self.max_len = self.bptt + max_deviation * std

        self.bptt_arr = []
        j = start
        while j < self._data.size - 2:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(self.max_len, max(self.min_len, int(np.random.normal(bptt, self.std))))
            self.bptt_arr.append(bptt)
            _seq_len = min(bptt, self._data.size - 1 - j)
            j += _seq_len
        self.len = len(self.bptt_arr)
        self.index = 0

    def __getitem__(self, item):
        bptt = self.bptt_arr[self.index]
        self.index += 1
        _seq_len = min(bptt, len(self._data) - 1 - item)

        end_idx = item + _seq_len
        beg_idx = max(0, item - self.ext_len)

        _data = self._data[beg_idx:end_idx]
        _target = self._data[item + 1:item + 1 + _seq_len]
        return _data, _target

    def __len__(self):
        return self.len


class AbstractDataset:
    def __init__(self, path, _dataset, *_args, **kwargs):
        super(AbstractDataset, self).__init__()
        self.path = path
        self.dataset = _dataset
        self.args = _args
        self.kwargs = kwargs

    def write(self):
        pass

    def get_train_generator(self):
        return self.train_generator

    def get_valid_generator(self):
        return self.valid_generator

    def get_test_generator(self):
        return self.test_generator


class Enwik8_Dataset(AbstractDataset):
    def __init__(self, path, _dataset, batch_size, tgt_len, *_args, ext_len=None, eval_tgt_len=None, varlen=False,
                 **kwargs):
        super(Enwik8_Dataset, self).__init__(path, _dataset, *_args, **kwargs)
        self.vocab = Vocab()
        self.vocab.count_file(os.path.join(path, 'train.txt'))
        self.vocab.count_file(os.path.join(path, 'valid.txt'))
        self.vocab.count_file(os.path.join(path, 'test.txt'))
        self.vocab.build_vocab()
        self.train = self.vocab.encode_file(
            os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        self.train_generator = getGenerator(self.train, batch_size, tgt_len, ext_len, varlen)
        self.valid_generator = getGenerator(self.valid, batch_size, eval_tgt_len, ext_len, varlen)
        self.test_generator = getGenerator(self.test, batch_size, eval_tgt_len, ext_len, varlen)


class Text8_Dataset(AbstractDataset):
    def __init__(self, path, _dataset, batch_size, tgt_len, *_args, ext_len=None, eval_tgt_len=None, varlen=False,
                 **kwargs):
        super(Text8_Dataset, self).__init__(path, _dataset, *_args, **kwargs)
        self.vocab = Vocab()
        self.vocab.count_file(os.path.join(path, 'train.txt'))
        self.vocab.count_file(os.path.join(path, 'valid.txt'))
        self.vocab.count_file(os.path.join(path, 'test.txt'))
        self.vocab.build_vocab()
        self.train = self.vocab.encode_file(
            os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        self.train_generator = getGenerator(self.train, batch_size, tgt_len, ext_len, varlen)
        self.valid_generator = getGenerator(self.valid, batch_size, eval_tgt_len, ext_len, varlen)
        self.test_generator = getGenerator(self.test, batch_size, eval_tgt_len, ext_len, varlen)


def getGenerator(_data, batch_size, tgt_len, ext_len=None, varlen=False, start=0, std=5, min_len=5, max_deviation=3):
    if varlen:
        return VariableGenerator(_data, batch_size, tgt_len, ext_len, start, std, min_len, max_deviation)
    return Generator(_data, batch_size, tgt_len, ext_len)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../../data/enwik8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='enwik8',
                        choices=['enwik8', 'text8'],
                        help='dataset name')
    parser.add_argument('--varlen', action='store_true', default=False,
                        help='use variable length')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=70,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    args = parser.parse_args()

    dataset = Enwik8_Dataset(args.datadir, args.dataset, args.batch_size, args.tgt_len, args.ext_len, args.varlen)
    train_dataset = GeneratorDataset(source=dataset.get_train_generator(), column_names=['data', 'target', 'seq_len'])

    for i, (data, target, seq_len) in enumerate(train_dataset.create_tuple_iterator()):
        print(str(i) + ' {}'.format(data))
