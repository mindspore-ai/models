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
"""tokenizer"""
import sys
from src.dataset import get_dataset
from  mindspore.dataset import text
import numpy as np

print(sys.path[0])

class CscTokenizer:
    def __init__(self, device_num=1, rank_id=0, fp="", max_seq_len=0, vocab_path='./src/bert-base-chinese-vocab.txt'):
        self.vocab_path = vocab_path
        self.tokenizer_op = self.set_tokenizer()
        self.fp = fp
        self.vocab = self.set_vocab()
        self.max_seq_len = max_seq_len
        self.device_num = device_num
        self.rank_id = rank_id

    def set_vocab(self):
        fr = open(self.vocab_path, mode='rb')
        vocab_byte_list = fr.readlines()
        vocab_byte_dict = {}
        count = 0
        for item in vocab_byte_list:
            if b'\n' in item:
                item = item[:-1]
            vocab_byte_dict[item] = count
            count = count + 1
        return vocab_byte_dict

    def set_tokenizer(self):
        vocab = text.Vocab.from_file(self.vocab_path)
        tokenizer_op = text.BertTokenizer(vocab=vocab)
        return tokenizer_op

    def convert2id(self, tokens_ndarray):
        tokens_list = list(tokens_ndarray)
        ids_list = []
        ids_list.append(101) #[CLS]
        for token in tokens_list:
            if token in self.vocab:
                ids_list.append(self.vocab[token])
            else:
                ids_list.append(100) #[UNK]
        ids_list.append(102) #[SEP]
        # pad to max_seq_len
        input_mask = [1 for i in range(len(ids_list))]
        while len(ids_list) < self.max_seq_len:
            ids_list.append(0)
            input_mask.append(0)
        assert len(ids_list) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        token_type_ids = [0 for i in range(self.max_seq_len)]
        ids_ndarray = np.array(ids_list, dtype=np.int32)
        input_mask_ndarray = np.array(input_mask, dtype=np.int32)
        token_type_ids_ndarray = np.array(token_type_ids, dtype=np.int32)
        return ids_ndarray, input_mask_ndarray, token_type_ids_ndarray

    def turn2int32(self, num_ndarray):
        return np.array(num_ndarray, dtype=np.int32)

    def get_worng_ids_ndarray(self, wrong_ids_byte_ndarray):
        i = 0
        wrong_ids_list = []
        wrong_ids_str = text.to_str(wrong_ids_byte_ndarray).tolist()
        while wrong_ids_str[i] != ']':
            if wrong_ids_str[i].isdigit():
                wrong_ids_list.append(int(wrong_ids_str[i]))
            i = i + 1
        wrong_ids_ndarray = np.array(wrong_ids_list)
        return wrong_ids_ndarray

    def convert2det_labels(self, wrong_ids_ndarray, tokens_ndarray):
        wrong_ids_list = wrong_ids_ndarray.tolist()
        tokens_list_byte = list(tokens_ndarray)
        # turn to string list
        tokens_list_str = []
        for token_byte in tokens_list_byte:
            tokens_list_str.append(text.to_str(token_byte))
        # create det_labels_list
        det_labels_list = [0 for i in range(len(tokens_list_str))]
        for idx in wrong_ids_list:
            margins = []
            for word in tokens_list_str[:idx]:
                if word == '[UNK]':
                    break
                if word.startswith('##'):
                    margins.append(len(word) - 3)
                else:
                    margins.append(len(word) - 1)
            margin = sum(margins)
            move = 0
            while (abs(move) < margin) or (idx + move >= len(tokens_list_str)) or \
            tokens_list_str[idx + move].startswith('##'):
                move -= 1
            det_labels_list[idx + move + 1] = 1
        # pad to max_seq_len
        while len(det_labels_list) < self.max_seq_len:
            det_labels_list.append(0)
        assert len(det_labels_list) == self.max_seq_len
        det_labels_ndarray = np.array(det_labels_list)
        return det_labels_ndarray

    def get_token_ids(self, batch_size):
        dataset = get_dataset(self.fp, vocab_path=self.vocab_path, device_num=self.device_num, rank_id=self.rank_id)
        dataset = dataset.map(operations=self.tokenizer_op, input_columns=['original_tokens'])
        dataset = dataset.map(operations=self.convert2id, input_columns=['original_tokens'],
                              output_columns=['original_tokens', 'original_tokens_mask', 'original_token_type_ids'])
        dataset = dataset.project(['wrong_ids', 'original_tokens', 'original_tokens_mask', 'correct_tokens',
                                   'original_token_type_ids'])
        dataset = dataset.map(operations=self.tokenizer_op, input_columns=['correct_tokens'])
        dataset = dataset.map(operations=self.convert2id, input_columns=['correct_tokens'],
                              output_columns=['correct_tokens', 'correct_tokens_mask', 'correct_token_type_ids'])
        dataset = dataset.project(['wrong_ids', 'original_tokens', 'original_tokens_mask', 'correct_tokens',
                                   'correct_tokens_mask', 'original_token_type_ids', 'correct_token_type_ids'])
        dataset = dataset.map(operations=self.turn2int32, input_columns=['wrong_ids'])
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        return dataset

if __name__ == '__main__':
    fpath = '../../dataset/csc/dev.json'
    demo = CscTokenizer(fpath)
    dataset1 = demo.get_token_ids()
    count1 = 0
    for data in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        count1 = count1 + 1
        if count1 > 3:
            break
