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
"""mlm_three"""
import random
import math
import numpy as np
from config import config
from toolz.sandbox import unzip
from .utils import pad_sequence
from .data import pad_tensors, pad_tensors_pos
from .data_three import DetectFeatTxtTokAudioFeatDataset, get_gather_index_three


def random_word_fix(tokens, vocab_range, mask, default_num=10):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    total_len = len(tokens)
    mask_len = math.ceil(default_num * 0.15)
    mask_num = random.sample([_ for _ in range(total_len)], mask_len)
    output_label = [-1 for _ in range(total_len)]
    for mask_index in mask_num:
        token = tokens[mask_index]
        tokens[mask_index] = mask
        output_label[mask_index] = token
    return tokens, output_label


def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def mask_all(tokens, mask):
    output_label = []

    for i, token in enumerate(tokens):
        tokens[i] = mask
        output_label.append(token)

    return tokens, output_label


def random_word_with_all(tokens, vocab_range, mask, rand_all_prob):
    prob = random.random()

    if prob < rand_all_prob:
        tokens, output_label = mask_all(tokens, mask)
    else:
        tokens, output_label = random_word(tokens, vocab_range, mask)

    return tokens, output_label


class MlmThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """
    class
    """

    def __init__(self, ids_path, txt_db, img_db, audio_db, rand_all_prob=0.3, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        self.rand_all_prob = rand_all_prob
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        ids = example['id']

        # text mask input and gt text
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'][:config.MAX_TEXT_LEN], self.use_mask_fix)

        # img input
        img_feat, img_pos_feat, _ = self._get_img_feat(example['img_fname'])

        audio_feat, _ = self._get_audio_feat(example['audio_fname'])

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        return (ids, input_ids, img_feat, img_pos_feat, attn_masks, txt_labels, audio_feat)

    def create_mlm_io(self, input_ids, use_mask_fix=False):
        """ create_mlm_io """
        # add mask
        if use_mask_fix:
            input_ids, txt_labels = random_word_fix(input_ids,
                                                    self.txt_db.v_range,
                                                    self.txt_db.mask)
        else:
            input_ids, txt_labels = random_word_with_all(input_ids,
                                                         self.txt_db.v_range,
                                                         self.txt_db.mask,
                                                         self.rand_all_prob)
        # add cls and sep
        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        txt_labels = np.array([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def mlmThree_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    :audio_feat   (n, audio_size, audio_dim)
    """
    (ids, input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, audio_feats) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1, max_lens=30)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

    # audio batches
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats, num_aus)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=90)

    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)

    # txt_labels bs,4   -1 888 999  -1
    txt_mask = np.stack((txt_labels != -1).nonzero(), 1)
    txt_label_mask = txt_labels[txt_labels != -1]

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels,
             'txt_mask': txt_mask,
             'txt_label_mask': txt_label_mask,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids,
             'ids': ids}
    return batch
