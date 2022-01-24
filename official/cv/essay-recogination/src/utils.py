#!/bin/bash
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

import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops.operations.comm_ops import ReduceOp

class CTCLabelConverter():
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['blank'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] if char in self.dict.keys() else 0 for char in text]

        return (Tensor(text, dtype=ms.int32), Tensor(length, dtype=ms.int32))

    def decode(self, text_index, length):
        text_index = self.revise(text_index)
        #convert text-index into text-label.
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i] < len(self.character):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

    def decodeOri(self, text_index, length):
        text_index = self.revise(text_index)
        text = ''
        for i in range(len(text_index)):
            if i % 45 == 0:
                text = ''
            if text_index[i] == 0:
                text = text + '--'
            elif self.character[text_index[i]] == ',':
                text = text + '，'
            elif self.character[text_index[i]] == '!':
                text = text + '！'
            elif self.character[text_index[i]] == '\n':
                text = text + '--'
            else:
                ch = self.character[text_index[i]]
                text = text + ch
            i = i + 1

        return text

    def revise(self, text_index):
        sq = np.array(text_index, dtype=np.int)
        sq = sq.reshape([90, 45])
        m = len(sq)
        n = len(sq[0])

        for i in range(m):
            cur_line = -1
            cur_j = n
            for j in range(n):
                if sq[i, j] > 0:
                    cur_line = i
                    cur_j = j
                    break

            for j in range(cur_j, n):
                if cur_line >= 0:
                    if cur_line + 1 < m:
                        if sq[cur_line + 1, j] != 0 and self.character[sq[cur_line + 1, j]] != '\n':
                            self.nested_block1(sq, cur_line, j, m)
                            tmp = sq[cur_line + 1, j]
                            sq[cur_line, j] = 0
                            cur_line = cur_line + 1
                            sq[cur_line, j] = 0
                            sq[i, j] = tmp

                    elif cur_line - 1 >= 0:
                        if sq[cur_line - 1, j] != 0 and self.character[sq[cur_line - 1, j]] != '\n':
                            self.nested_block2(sq, cur_line, j)

                            tmp = sq[cur_line - 1, j]
                            sq[cur_line, j] = 0
                            cur_line = cur_line - 1
                            sq[cur_line, j] = 0
                            sq[i, j] = tmp

                    if sq[cur_line, j] != 0:
                        tmp = sq[cur_line, j]
                        sq[cur_line, j] = 0
                        sq[i, j] = tmp

        text_index = sq.reshape([45*90])
        return text_index

    def nested_block1(self, s, cur, index, m):
        if cur + 2 < m and s[cur + 2, index] != 0:
            s[cur + 2, index] = 0

    def nested_block2(self, s, cur, index):
        if cur - 2 >= 0 and s[cur - 2, index] != 0:
            s[cur - 2, index] = 0


class Averager():
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

#Horovod: average metrics from distributed training.
class Metric():
    def __init__(self, parO, name=''):
        self.name = name
        self.sum = ms.Tensor(0., ms.float32)
        self.n = ms.Tensor(0., ms.float32)
        self.pO = parO

    def update(self, val):
        if self.pO.HVD:
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name).double()
        elif self.pO.DDP:
            rt = val.clone()
            reduce = ops.AllReduce(op=ReduceOp.SUM)
            rt = reduce(rt)
            rt /= dist.get_world_size()
            self.sum += rt.detach().cpu().double()
        elif self.pO.DP:
            self.sum += val.detach().double()

        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n.double()
