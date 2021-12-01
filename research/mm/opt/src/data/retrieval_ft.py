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
"""retrieval_ft"""
import random
import numpy as np
from cytoolz import concat
from config import config
from toolz.sandbox import unzip
from .data_three import DetectFeatTxtTokTwoDataset, get_ids_three
from .data import pad_tensors, pad_tensors_pos, get_gather_index
from .utils import pad_sequence



def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class ItmFlickrRankDataset(DetectFeatTxtTokTwoDataset):
    """
    ItmFlickrRankDataset
    """
    def __init__(self, ids_path, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, \
            "ItmRankDataset need at least 1 negative sample"
        super().__init__(ids_path, txt_db, img_db, use_video=False)

        self.ids = get_ids_three(ids_path)

        assert neg_sample_size > 0
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.ids[i]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_img_ids = sample_negative(
            self.ids, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, [gt_txt_id], self.neg_sample_size)
        id_pairs.extend([(gt_txt_id, neg_img_id)
                         for neg_img_id in neg_sample_img_ids] +
                        [(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + 2 * self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        """
        collect_inputs
        """
        # create input features
        inputs = []
        for txt_id, img_id in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids'][:config.MAX_TEXT_LEN]
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            img_feat, img_pos_feat, _ = self._get_img_feat(img_id)
            # mask
            attn_masks = np.ones(len(input_ids) + img_feat.shape[0], dtype=np.int64)
            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks))

        return inputs


def itm_rank_collate(inputs):
    """
    itm_rank_collate
    """
    (input_ids, img_feats, img_pos_feats, attn_masks,
     ) = map(list, unzip(concat(i for i in inputs)))
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=90)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.shape
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'sample_size': sample_size}
    return batch
