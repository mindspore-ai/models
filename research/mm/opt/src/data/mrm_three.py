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
"""mrm_three"""
import random
import math
import numpy as np
from toolz.sandbox import unzip
from .utils import pad_sequence, masked_fill
from .data import pad_tensors, pad_tensors_pos
from .data_three import DetectFeatTxtTokAudioFeatDataset, get_gather_index_three


def _get_img_mask(mask_prob, num_bb):
    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    img_mask = np.array(img_mask)
    return img_mask


def _get_img_mask_fix(mask_prob, num_bb, default_num=10):
    mask_len = math.ceil(default_num * mask_prob)
    mask_num = random.sample(range(num_bb), mask_len)
    img_mask = [False] * num_bb
    for i in mask_num:
        img_mask[i] = True
    img_mask = np.array(img_mask)
    return img_mask


def _get_img_mask_all(num_bb):
    img_mask = np.ones(num_bb).astype(np.bool_)
    return img_mask


def _get_img_mask_with_all(mask_prob, num_bb):
    prob = random.random()
    if prob < 0.3:
        img_mask = _get_img_mask_all(num_bb)
    else:
        img_mask = _get_img_mask(mask_prob, num_bb)
    return img_mask


def _get_img_tgt_mask_three(img_mask, txt_len, audio_len):
    z = np.zeros(txt_len, dtype=np.bool_)
    a = np.zeros(audio_len, dtype=np.bool_)
    img_mask_tgt = np.concatenate([z, img_mask, a], axis=0)
    return img_mask_tgt


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = np.broadcast_to(np.expand_dims(img_masks, -1), img_feat.shape)  # (n, m, d)
    feat_dim = img_feat.shape[-1]
    feat_targets = np.ascontiguousarray(img_feat[img_masks_ext]).reshape(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = np.broadcast_to(np.expand_dims(img_masks, -1), img_feat.shape)

    img_feat_masked = masked_fill(img_feat, img_masks_ext, 0)
    return img_feat_masked


def _get_targets(img_masks, img_soft_label):
    soft_label_dim = img_soft_label.shape[-1]
    img_masks_ext_for_label = np.broadcast_to(np.expand_dims(img_masks, -1), img_soft_label.shape)
    label_targets = np.ascontiguousarray(img_soft_label[img_masks_ext_for_label]).reshape(
        -1, soft_label_dim)
    return label_targets


class MrfrThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """
    MrfrThreeDataset
    """
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
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask     : (num_bb, ) between {0, 1}
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

        if self.use_mask_fix:
            img_mask = _get_img_mask_fix(self.mask_prob, num_bb)
        else:
            img_mask = _get_img_mask_with_all(self.mask_prob, num_bb)

        audio_feat, audio_len = self._get_audio_feat(example['audio_fname'])

        img_mask_tgt = _get_img_tgt_mask_three(img_mask, len(input_ids), audio_len)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)
        return (input_ids, img_feat, img_pos_feat,
                attn_masks, img_mask, img_mask_tgt,
                audio_feat)


def mrfrThree_collate(inputs):
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
    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts, audio_feats) = map(list, unzip(inputs))

    txt_lens = [i.shape[0] for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0, max_lens=30)

    # audio features
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=30)
    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)

    img_mask_tgt_mask = np.stack((img_mask_tgt).nonzero(), 1)
    neg_index = np.stack(img_masks.reshape(-1).nonzero(), 1).squeeze()
    n_negatives = 10
    neg = np.zeros([neg_index.shape[0], n_negatives, img_feat.shape[-1]])
    neg_feat = img_feat.reshape(-1, img_feat.shape[-1])
    for i in range(neg_index.shape[0]):
        left = (neg_index[i] - n_negatives // 2).item()
        right = (neg_index[i] + n_negatives // 2 + 1).item()
        if left < 0:
            negs = np.concatenate([neg_feat[0:neg_index[i]], neg_feat[neg_index[i] + 1: n_negatives + 1]], axis=0)
        elif right > neg_feat.shape[0]:
            negs = np.concatenate([neg_feat[neg_feat.shape[0] - n_negatives - 1:neg_index[i]],
                                   neg_feat[neg_index[i] + 1: neg_feat.shape[0]]], axis=0)
        else:
            negs = np.concatenate([neg_feat[left:neg_index[i]], neg_feat[neg_index[i] + 1: right]], axis=0)

        neg[i, :, :] = np.expand_dims(negs, 0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'img_mask_tgt_mask': img_mask_tgt_mask,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids,
             'mr_neg_index': np.expand_dims(neg_index, -1),
             'mr_neg_sample': neg.astype(np.float32)
             }
    return batch


# Masked Region Classification
class MrcThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """
    MrcThreeDataset
    """
    def __init__(self, ids_path, mask_prob, txt_db, img_db, audio_db, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db)
        self.mask_prob = mask_prob
        self.use_mask_fix = use_mask_fix

    def _get_img_feat(self, idx):

        att_feat, pred_boxes, _, pred_classes = self.img_db[idx]
        img_feat = np.array(att_feat)

        img_soft_label = np.expand_dims(np.array(pred_classes), -1)

        img_bb = pred_boxes
        num_bb = att_feat.shape[0]

        return img_feat, img_bb, img_soft_label, num_bb

    def box2feat(self, bb):
        img_bb = np.concatenate([bb, bb[:, 4:5] * bb[:, 5:]], axis=-1)
        return img_bb

    def __getitem__(self, i):
        example = super().__getitem__(i)

        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(example["img_fname"])

        # image input features
        if self.use_mask_fix:
            img_mask = _get_img_mask_fix(self.mask_prob, num_bb)
        else:
            img_mask = _get_img_mask_with_all(self.mask_prob, num_bb)

        # text input
        input_ids = example['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:8]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # audio features
        audio_feat, num_audio = self._get_audio_feat(example['audio_fname'])

        img_mask_tgt = _get_img_tgt_mask_three(img_mask, len(input_ids), num_audio)

        attn_masks = np.ones(len(input_ids) + num_bb + num_audio, dtype=np.int64)
        return (input_ids, img_feat, img_pos_feat,
                img_soft_labels, attn_masks, img_mask, img_mask_tgt, audio_feat)


def mrcThree_collate(inputs):
    """
    mrcThree_collate
    """
    (input_ids, img_feats, img_pos_feats, img_soft_labels,
     attn_masks, img_masks, img_mask_tgts, audio_feats) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    img_soft_label = pad_tensors(img_soft_labels, num_bbs, max_len=-1)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    label_targets = _get_targets(img_masks, img_soft_label)

    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0, max_lens=30)

    # audio batches
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=30)

    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)

    img_mask_tgt_mask = np.stack((img_mask_tgt).nonzero(), 1)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'img_mask_tgt_mask': img_mask_tgt_mask,
             'label_targets': label_targets,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids}
    return batch
