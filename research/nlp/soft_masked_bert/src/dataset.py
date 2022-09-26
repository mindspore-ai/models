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

from src.utils import get_abs_path, load_json
import mindspore.dataset as ds
import mindspore.dataset.text as text
from tqdm import tqdm

def get_dataset(fp, vocab_path='./src/bert-base-chinese-vocab.txt', max_seq_len=512, shffle=True, \
                workers_num=1, device_num=1, rank_id=0):
    data = load_json(fp)
    original_text_list = []
    original_tokens_list = []
    wrong_ids_list = []
    correct_text_list = []
    correct_tokens_list = []
    det_label_list = []
    # tokenizer
    vocab = text.Vocab.from_file(vocab_path)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    for item in tqdm(data):
        original_text_list.append(item['original_text'])
        original_tokens_list.append(item['original_text'])
        wrong_ids_list.append(str(item['wrong_ids']))
        encoded_text = tokenizer_op(item['correct_text'])
        det_label = [0 for i in range(max_seq_len)]
        for idx in item['wrong_ids']:
            margins = []
            for word in encoded_text[:idx]:
                if word == '[UNK]':
                    break
                if word.startswith('##'):
                    margins.append(len(word) - 3)
                else:
                    margins.append(len(word) - 1)
            margin = sum(margins)
            move = 0
            while (abs(move) < margin) or (idx + move >= len(encoded_text)) or encoded_text[idx + move].startswith(
                    '##'):
                move -= 1
            det_label[idx + move + 1] = 1
        det_label_list.append(det_label)
        correct_text_list.append(item['correct_text'])
        correct_tokens_list.append(item['correct_text'])
    if device_num > 1:
        dataset = ds.NumpySlicesDataset(data=(original_text_list, det_label_list, correct_text_list),
                                        column_names=['original_tokens', 'wrong_ids', 'correct_tokens'],
                                        num_shards=device_num, shard_id=rank_id)
    else:
        dataset = ds.NumpySlicesDataset(data=(original_text_list, det_label_list, correct_text_list),
                                        column_names=['original_tokens', 'wrong_ids', 'correct_tokens'])
    return dataset

def make_datasets(cfg, get_loader_fn, tokenizer, **kwargs):
    if cfg.DATASETS.TRAIN == '':
        train_dataset = None
    else:
        train_dataset = get_loader_fn(get_abs_path(cfg.DATASETS.TRAIN), \
                                     batch_size=cfg.SOLVER.BATCH_SIZE, \
                                     shuffle=True, \
                                     num_workers=cfg.DATALOADER.NUM_WORKERS, \
                                     tokenizer=tokenizer, **kwargs)
    if cfg.DATASETS.VALID == '':
        valid_dataset = None
    else:
        valid_dataset = get_loader_fn(get_abs_path(cfg.DATASETS.VALID), \
                                     batch_size=cfg.TEST.BATCH_SIZE, \
                                     shuffle=False, \
                                     num_workers=cfg.DATALOADER.NUM_WORKERS, \
                                     tokenizer=tokenizer, **kwargs)
    if cfg.DATASETS.TEST == '':
        test_dataset = None
    else:
        test_dataset = get_loader_fn(get_abs_path(cfg.DATASETS.TEST), \
                                    batch_size=cfg.TEST.BATCH_SIZE, \
                                    shuffle=False, \
                                    num_workers=cfg.DATALOADER.NUM_WORKERS, \
                                    tokenizer=tokenizer, **kwargs)
    return train_dataset, valid_dataset, test_dataset
