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
Dataset interfaces
"""

from contextlib import contextmanager
from mindspore.communication.management import get_group_size, get_rank

import numpy as np
import lmdb
from lz4.frame import compress, decompress

import msgpack
import msgpack_numpy

msgpack_numpy.patch()


def _fp16_to_fp32(feat_dict):
    """ _fp16_to_fp32 """
    out = {k: arr.astype(np.float32) if arr.dtype == np.float16 else arr for k, arr in feat_dict.items()}
    return out


def compute_num_bb(confs, conf_th, min_bb, max_bb):
    """ compute_num_bb """

    num_bb = max(min_bb, (confs > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return num_bb


def _check_distributed():
    """ _check_distributed """

    try:
        dist = get_group_size() != get_rank()
    except ValueError:
        # not using horovod
        dist = False
    return dist


@contextmanager
def open_lmdb(db_dir, readonly=False):
    """ open_lmdb """

    db = TxtLmdb(db_dir, readonly)
    try:
        yield db
    finally:
        del db


class TxtLmdb():
    """ TxtLmdb """

    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 readahead=not _check_distributed(), lock=False)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024 ** 4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret


# pad tensors
def pad_tensors(tensors, lens=None, pad=0, max_len=30):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    if max_len == -1:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = np.zeros((bs, max_len, hid), dtype=dtype)
    if pad:
        output.fill(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l, ...] = t
    return output


def pad_tensors_pos(tensors, lens, feat, max_len=30):
    """ pad_tensors_pos """
    if tensors is None or tensors[0] is None:
        return np.expand_dims(np.arange(0, feat.shape[1], dtype=np.int64), 0)
    return pad_tensors(tensors, lens, max_len=max_len)


def get_gather_index_three(txt_lens, num_bbs, num_aus, batch_size, max_len, max_bb, out_size):
    """ get_gather_index_three """

    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    for i, (tl, nbb, nau) in enumerate(zip(txt_lens, num_bbs, num_aus)):
        gather_index[i, tl:tl + nbb] = np.arange(max_len, max_len + nbb, dtype=np.int64)
        # 32, 144 - 121
        gather_index[i, tl + nbb:tl + nbb + nau] = np.arange(max_len + max_bb, max_len + max_bb + nau, dtype=np.int64)

    return gather_index


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    """ get_gather_index """

    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index[i, tl:tl + nbb] = np.arange(max_len, max_len + nbb, dtype=np.int64)

    return gather_index
