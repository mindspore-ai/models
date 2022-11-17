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
dataset used in DGU.
"""

import os
from typing import List
import numpy as np

# The input data bigin with '[CLS]', using '[SEP]' split conversation content(
# Previous part, current part, following part, etc.). If there are multiple
# conversation in split part, using 'INNER_SEP' to further split.
INNER_SEP = '[unused0]'

class Tuple():
    """
    apply the functions to  the corresponding input fields.
    """
    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert args, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn,) + args
        for i, ele_fn in enumerate(self._fn):
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                i, str(type(ele_fn)))

    def __call__(self, data):

        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contain' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            result = ele_fn([ele[i] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)


class Pad():
    """
     pad the data with given value
    """
    def __init__(self,
                 pad_val=0,
                 axis=0,
                 ret_length=None,
                 dtype=None,
                 pad_right=True):
        self._pad_val = pad_val
        self._axis = axis
        self._ret_length = ret_length
        self._dtype = dtype
        self._pad_right = pad_right

    def __call__(self, data):
        arrs = [np.asarray(ele) for ele in data]
        original_length = [ele.shape[self._axis] for ele in arrs]
        max_size = max(original_length)
        ret_shape = list(arrs[0].shape)
        ret_shape[self._axis] = max_size
        ret_shape = (len(arrs),) + tuple(ret_shape)
        ret = np.full(
            shape=ret_shape,
            fill_value=self._pad_val,
            dtype=arrs[0].dtype if self._dtype is None else self._dtype)
        for i, arr in enumerate(arrs):
            if arr.shape[self._axis] == max_size:
                ret[i] = arr
            else:
                slices = [slice(None) for _ in range(arr.ndim)]
                if self._pad_right:
                    slices[self._axis] = slice(0, arr.shape[self._axis])
                else:
                    slices[self._axis] = slice(max_size - arr.shape[self._axis],
                                               max_size)

                if slices[self._axis].start != slices[self._axis].stop:
                    slices = [slice(i, i + 1)] + slices
                    ret[tuple(slices)] = arr
        if self._ret_length:
            return ret, np.asarray(
                original_length,
                dtype="int32") if self._ret_length else np.asarray(
                    original_length, self._ret_length)
        return ret


class Stack():
    """
    Stack the input data
    """

    def __init__(self, axis=0, dtype=None):
        self._axis = axis
        self._dtype = dtype

    def __call__(self, data):
        data = np.stack(
            data,
            axis=self._axis).astype(self._dtype) if self._dtype else np.stack(
                data, axis=self._axis)
        return data


class Dataset():
    """ Dataset base class """
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class " \
                                  "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class " \
                                  "{}".format('__len__', self.__class__.__name__))


def get_label_map(label_list):
    """ Create label maps """
    label_map = {}
    for (i, l) in enumerate(label_list):
        label_map[l] = i
    return label_map


class ATIS_DID(Dataset):
    """
    The dataset ATIS_ID is using in task Dialogue Intent Detection.
    The source dataset is ATIS(Airline Travel Information System). See detail at
    https://www.kaggle.com/siddhadev/ms-cntk-atis
    """
    LABEL_MAP = get_label_map([str(i) for i in range(26)])

    def __init__(self, data_dir, mode='test'):
        super(ATIS_DID, self).__init__()
        self._data_dir = data_dir
        self._mode = mode
        self.read_data()

    def read_data(self):
        """read data from file"""
        if self._mode == 'train':
            data_path = os.path.join(self._data_dir, 'train.txt')
        elif self._mode == 'dev':
            data_path = os.path.join(self._data_dir, 'dev.txt')
        elif self._mode == 'test':
            data_path = os.path.join(self._data_dir, 'test.txt')
        elif self._mode == 'infer':
            data_path = os.path.join(self._data_dir, 'infer.txt')
        self.data = []
        with open(data_path, 'r', encoding='utf8') as fin:
            for line in fin:
                if not line:
                    continue
                arr = line.rstrip('\n').split('\t')
                if len(arr) != 2:
                    print('Data format error: %s' % '\t'.join(arr))
                    print(
                        'Data row should contains two parts: label\tconversation_content.'
                    )
                    continue
                label = arr[0]
                text = arr[1]
                self.data.append([label, text])

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        label, text = example
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[len(tokens) - max_seq_length + 2:]
        tokens_, segment_ids = [], []
        tokens_.append("[CLS]")
        for token in tokens:
            tokens_.append(token)
        tokens_.append("[SEP]")
        tokens = tokens_
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label = np.array([cls.get_label(label)], dtype='int64')
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        return input_ids, input_mask, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def read_da_data(data_dir, mode):
    """read data from file"""
    def _concat_dialogues(examples):
        """concat multi turns dialogues"""
        new_examples = []
        example_len = len(examples)
        for i in range(example_len):
            label, caller, text = examples[i]
            cur_txt = "%s : %s" % (caller, text)
            pre_txt = [
                "%s : %s" % (item[1], item[2])
                for item in examples[max(0, i - 5):i]
            ]
            suf_txt = [
                "%s : %s" % (item[1], item[2])
                for item in examples[i + 1:min(len(examples), i + 3)]
            ]
            sample = [label, pre_txt, cur_txt, suf_txt]
            new_examples.append(sample)
        return new_examples

    if mode == 'train':
        data_path = os.path.join(data_dir, 'train.txt')
    elif mode == 'dev':
        data_path = os.path.join(data_dir, 'dev.txt')
    elif mode == 'test':
        data_path = os.path.join(data_dir, 'test.txt')
    elif mode == 'infer':
        data_path = os.path.join(data_dir, 'infer.txt')
    data = []
    with open(data_path, 'r', encoding='utf8') as fin:
        pre_idx = -1
        examples = []
        for line in fin:
            if not line:
                continue
            arr = line.rstrip('\n').split('\t')
            if len(arr) != 4:
                print('Data format error: %s' % '\t'.join(arr))
                print(
                    'Data row should contains four parts: id\tlabel\tcaller\tconversation_content.'
                )
                continue
            idx, label, caller, text = arr
            if idx != pre_idx:
                if idx != 0:
                    examples = _concat_dialogues(examples)
                    data.extend(examples)
                    examples = []
                pre_idx = idx
            examples.append((label, caller, text))
        if examples:
            examples = _concat_dialogues(examples)
            data.extend(examples)
    return data


def truncate_and_concat(pre_txt: List[str],
                        cur_txt: str,
                        suf_txt: List[str],
                        tokenizer,
                        max_seq_length,
                        max_len_of_cur_text):
    """concat data"""
    cur_tokens = tokenizer.tokenize(cur_txt)
    cur_tokens = cur_tokens[:min(max_len_of_cur_text, len(cur_tokens))]
    pre_tokens = []
    for text in pre_txt:
        pre_tokens.extend(tokenizer.tokenize(text))
        pre_tokens.append(INNER_SEP)
    pre_tokens = pre_tokens[:-1]
    suf_tokens = []
    for text in suf_txt:
        suf_tokens.extend(tokenizer.tokenize(text))
        suf_tokens.append(INNER_SEP)
    suf_tokens = suf_tokens[:-1]
    if len(cur_tokens) + len(pre_tokens) + len(suf_tokens) > max_seq_length - 4:
        left_num = max_seq_length - 4 - len(cur_tokens)
        if len(pre_tokens) > len(suf_tokens):
            suf_num = int(left_num / 2)
            suf_tokens = suf_tokens[:suf_num]
            pre_num = left_num - len(suf_tokens)
            pre_tokens = pre_tokens[max(0, len(pre_tokens) - pre_num):]
        else:
            pre_num = int(left_num / 2)
            pre_tokens = pre_tokens[max(0, len(pre_tokens) - pre_num):]
            suf_num = left_num - len(pre_tokens)
            suf_tokens = suf_tokens[:suf_num]
    tokens, segment_ids = [], []
    tokens.append("[CLS]")
    for token in pre_tokens:
        tokens.append(token)
    tokens.append("[SEP]")
    segment_ids.extend([0] * len(tokens))
    for token in cur_tokens:
        tokens.append(token)
    tokens.append("[SEP]")
    segment_ids.extend([1] * (len(cur_tokens) + 1))
    if suf_tokens:
        for token in suf_tokens:
            tokens.append(token)
        tokens.append("[SEP]")
        segment_ids.extend([0] * (len(suf_tokens) + 1))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return input_ids, input_mask, segment_ids


class MRDA(Dataset):
    """
    The dataset MRDA is using in task Dialogue Act.
    The source dataset is MRDA(Meeting Recorder Dialogue Act). See detail at
    https://www.aclweb.org/anthology/W04-2319.pdf
    """
    MAX_LEN_OF_CUR_TEXT = 50
    LABEL_MAP = get_label_map([str(i) for i in range(5)])

    def __init__(self, data_dir, mode='test'):
        super(MRDA, self).__init__()
        self.data = read_da_data(data_dir, mode)

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        label, pre_txt, cur_txt, suf_txt = example
        label = np.array([cls.get_label(label)], dtype='int64')
        input_ids, input_mask, segment_ids = truncate_and_concat(pre_txt, cur_txt, suf_txt, \
            tokenizer, max_seq_length, cls.MAX_LEN_OF_CUR_TEXT)
        return input_ids, input_mask, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SwDA(Dataset):
    """
    The dataset SwDA is using in task Dialogue Act.
    The source dataset is SwDA(Switchboard Dialog Act). See detail at
    http://compprag.christopherpotts.net/swda.html
    """
    MAX_LEN_OF_CUR_TEXT = 50
    LABEL_MAP = get_label_map([str(i) for i in range(42)])

    def __init__(self, data_dir, mode='test'):
        super(SwDA, self).__init__()
        self.data = read_da_data(data_dir, mode)

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        label, pre_txt, cur_txt, suf_txt = example
        label = np.array([cls.get_label(label)], dtype='int64')
        input_ids, input_mask, segment_ids = truncate_and_concat(pre_txt, cur_txt, suf_txt, \
            tokenizer, max_seq_length, cls.MAX_LEN_OF_CUR_TEXT)
        return input_ids, input_mask, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
