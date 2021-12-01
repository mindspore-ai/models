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
"""mam_three"""
import random
import math
import numpy as np
from toolz.sandbox import unzip
from .data import pad_tensors, pad_tensors_pos
from .data_three import DetectFeatTxtTokAudioFeatDataset, \
    get_gather_index_three

from .utils import pad_sequence


def _get_audio_mask_fix(mask_prob, num_audio, default_num=10):
    mask_len = math.ceil(default_num * mask_prob)
    mask_num = random.sample(range(num_audio), mask_len)
    audio_mask = [False] * num_audio
    for i in mask_num:
        audio_mask[i] = True
    audio_mask = np.array(audio_mask)
    return audio_mask


def _get_audio_mask(mask_prob, num_audio):
    audio_mask = [random.random() < mask_prob for _ in range(num_audio)]
    if not any(audio_mask):
        # at least mask 1
        audio_mask[random.choice(range(num_audio))] = True
    audio_mask = np.array(audio_mask)
    return audio_mask


def _get_audio_mask_all(num_audio):
    audio_mask = np.ones((num_audio), dtype=bool)
    return audio_mask


def _get_audio_mask_with_all(mask_prob, num_audio):
    prob = random.random()
    if prob < 0.3:
        img_mask = _get_audio_mask_all(num_audio)
    else:
        img_mask = _get_audio_mask(mask_prob, num_audio)
    return img_mask


def _get_audio_tgt_mask_text_audio(audio_mask, txt_len):
    z = np.zeros(txt_len, dtype=np.int64)
    audio_mask_tgt = np.concatenate((z, audio_mask), axis=0)
    return audio_mask_tgt


def _get_audio_tgt_mask(audio_mask, txt_len, img_len):
    z = np.zeros(txt_len + img_len, dtype=np.int64)
    audio_mask_tgt = np.concatenate((z, audio_mask), axis=0)
    return audio_mask_tgt


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = np.broadcast_to(np.expand_dims(img_masks, -1), img_feat.shape)
    feat_dim = img_feat.shape[-1]
    feat_targets = np.ascontiguousarray(img_feat[img_masks_ext]).reshape(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _get_audio_feat_target(audio_feat, audio_masks):
    audio_masks_ext = np.broadcast_to(np.expand_dims(audio_masks, -1), audio_feat.shape)
    feat_dim = audio_feat.shape[-1]
    audio_feat_targets = np.ascontiguousarray(audio_feat[audio_masks_ext]).reshape(
        -1, feat_dim)  # (s, d)
    return audio_feat_targets


def _mask_img_feat(img_feat, img_masks):
    if img_masks is None or img_masks[0] is None:
        return img_feat
    img_masks_ext = np.broadcast_to(np.expand_dims(img_masks, -1), img_feat.shape)
    img_feat[img_masks_ext] = 0
    return img_feat


def _mask_audio_feat(audio_feat, audio_masks):
    if audio_masks is None or audio_masks[0] is None:
        return audio_feat
    audio_masks_ext = np.broadcast_to(np.expand_dims(audio_masks, -1), audio_feat.shape)
    audio_feat[audio_masks_ext] = 0
    return audio_feat


class MafrThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """MafrThreeDataset"""
    def __init__(self, mask_prob, ids_path, txt_db, img_db, audio_db, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        self.mask_prob = mask_prob
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb + num_audio, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - audio_mask     : (num_audio, ) between {0, 1}
        - audio_feat
        """
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:28]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image input features
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        audio_feat, num_audio = self._get_audio_feat(example['audio_fname'])
        if self.use_mask_fix:
            audio_mask = _get_audio_mask_fix(self.mask_prob, num_audio)
        else:
            audio_mask = _get_audio_mask_with_all(self.mask_prob, num_audio)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        audio_mask_tgt = _get_audio_tgt_mask(audio_mask, len(input_ids), num_bb)

        return (input_ids, img_feat, img_pos_feat,
                attn_masks,
                audio_feat, audio_mask, audio_mask_tgt)


def mafrThree_collate(inputs):
    """
    Return:
    - input_ids    : (n, max_L), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
    - position_ids : (n, max_L)
    - txt_lens     : list of [input_len]
    - img_feat     : (n, max_num_bb, d)
    - img_pos_feat : (n, max_num_bb, 7)
    - num_bbs      : list of [num_bb]
    - attn_masks   : (n, max_{L + num_bb}), ie., [1, 1, ..., 0, 0, 1, 1]
    - img_masks    : (n, max_num_bb) between {0, 1}
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, audio_feats, audio_masks, audio_mask_tgts) = map(list,
                                                                                                       unzip(inputs))

    txt_lens = [i.shape[0] for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

    # audio features
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats)
    audio_masks = pad_sequence(audio_masks, batch_first=True, padding_value=0)
    audio_feat_targets = _get_audio_feat_target(audio_feat, audio_masks)
    audio_feat = _mask_audio_feat(audio_feat, audio_masks)
    audio_mask_tgt = pad_sequence(audio_mask_tgts,
                                  batch_first=True, padding_value=0, max_lens=-1)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)
    audio_mask_tgt_mask = np.stack((audio_mask_tgt).nonzero(), 1)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=90)
    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)
    neg_index = (audio_masks.reshape((-1)).nonzero())[0]
    n_negatives = 10
    neg = np.zeros([neg_index.shape[0], n_negatives, audio_feat.shape[-1]])
    neg_feat = audio_feat.reshape(-1, audio_feat.shape[-1])
    for i in range(neg_index.shape[0]):
        left = (neg_index[i] - n_negatives // 2).item()
        right = (neg_index[i] + n_negatives // 2 + 1).item()
        if left < 0:
            negs = np.concatenate((neg_feat[0:neg_index[i]], neg_feat[neg_index[i] + 1: n_negatives + 1]), axis=0)
        elif right > neg_feat.shape[0]:
            negs = np.concatenate((neg_feat[neg_feat.shape[0] - n_negatives - 1:neg_index[i]],
                                   neg_feat[neg_index[i] + 1: neg_feat.shape[0]]), axis=0)
        else:
            negs = np.concatenate((neg_feat[left:neg_index[i]], neg_feat[neg_index[i] + 1: right]), axis=0)

        neg[i] = negs

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'audio_feat_targets': audio_feat_targets,
             'audio_masks': audio_masks,
             'audio_mask_tgt': audio_mask_tgt,
             'audio_mask_tgt_mask': audio_mask_tgt_mask,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids,
             'ma_neg_index': neg_index,
             'ma_neg_sample': neg.astype(np.float32)}
    return batch
