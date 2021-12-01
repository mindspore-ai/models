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
MLM datasets
"""
import random
import json
import os
from config import config
from toolz.sandbox import unzip
import numpy as np

from .data import pad_tensors, pad_tensors_pos
from .data_three import (DetectFeatTxtTokAudioFeatDataset, get_gather_index_three)
from .mlm_three import random_word_with_all, random_word_fix
from .mrm_three import _get_img_mask, _get_img_mask_all
from .mam_three import _get_audio_mask, _get_audio_mask_all
from .id_three import _mask_img_feat, _mask_audio_feat
from .utils import pad_sequence, pad_sequence_


class TdThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """TdThreeDataset"""
    def __init__(self, ids_path, txt_db, img_db, audio_db, mask_type=-1, use_video=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video=use_video)
        self.mask_type = mask_type
        self.use_video = use_video
        self.rand_all_prob = 1.1

    def create_mlm_io(self, input_ids, use_mask_fix=False):
        """create """
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

    def mask_img_audio(self, num_bb, num_audio):
        """mask image tokens and audio tokens"""
        prob = random.random()
        if prob < 0.3:
            img_mask = _get_img_mask_all(num_bb)
            audio_mask = _get_audio_mask(0.15, num_audio)
            mask_type = 0
        elif prob < 0.6:
            img_mask = _get_img_mask(0.15, num_bb)
            audio_mask = _get_audio_mask_all(num_audio)
            mask_type = 1
        else:
            img_mask = _get_img_mask(0.15, num_bb)
            audio_mask = _get_audio_mask(0.15, num_audio)
            mask_type = 2

        return img_mask, audio_mask, mask_type

    def mask_img_audio_with_type(self, num_bb, num_audio):
        """
            mask image tokens and audio tokens with type:
            - type = -1 : random mask
            - type = 0  : image
            - type = 1  : audio
            - type = 2  : image+audio
        """
        if self.mask_type == 0:
            img_mask = None
            audio_mask = _get_audio_mask_all(num_audio)
            mask_type = 0
        elif self.mask_type == 1:
            img_mask = _get_img_mask_all(num_bb)
            audio_mask = None
            mask_type = 1
        elif self.mask_type == 2:
            img_mask = None
            audio_mask = None
            mask_type = 2
        else:
            raise Exception("Error Mask Type {}".format(self.mask_type))

        return img_mask, audio_mask, mask_type

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
        input_ids, _ = self.create_mlm_io(example['input_ids'][:config.MAX_TEXT_LEN], False)
        txt_inputs, txt_gts, txt_masks = self._get_txt_token(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

        audio_feat, num_audio = self._get_audio_feat(example['audio_fname'])

        if self.mask_type >= 0:
            img_mask, audio_mask, mask_type = self.mask_img_audio_with_type(num_bb, num_audio)
        else:
            img_mask, audio_mask, mask_type = self.mask_img_audio(num_bb, num_audio)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        return (ids, input_ids, img_feat, img_pos_feat, attn_masks, audio_feat,
                txt_inputs, txt_gts, txt_masks, img_mask, audio_mask, mask_type)


def tdThree_collate(inputs):
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
    (ids, input_ids, img_feats, img_pos_feats, attn_masks, audio_feats,
     txt_inputs, txt_gts, txt_masks, img_masks, audio_masks, mask_types) = map(list, unzip(inputs))

    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)
    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)
    img_masks = pad_sequence_(img_masks, batch_first=True, padding_value=0)
    img_feat = _mask_img_feat(img_feat, img_masks)

    # audio batches
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats, num_aus)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)
    audio_masks = pad_sequence_(audio_masks, batch_first=True, padding_value=0)
    audio_feat = _mask_audio_feat(audio_feat, audio_masks)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=90)

    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]

    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)

    # txt decoder
    txt_inputs = pad_sequence(txt_inputs, batch_first=True, padding_value=0, max_lens=90)
    txt_gts = pad_sequence(txt_gts, batch_first=True, padding_value=0, max_lens=90)
    txt_masks = pad_sequence(txt_masks, batch_first=True, padding_value=0, max_lens=90)
    txt_pos = np.expand_dims(np.arange(0, txt_inputs.shape[1], dtype=np.int64
                                       ), 0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': None,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids,
             'txt_inputs': txt_inputs,
             'txt_gts': txt_gts,
             'txt_masks': txt_masks,
             'txt_pos': txt_pos,
             'ids': ids,
             'img_masks': img_masks,
             'audio_masks': audio_masks,
             'mask_types': mask_types}
    return batch


class TdOneDataset:
    """TdOneDataset"""
    def __init__(self, ids_path, img_dir_npz):

        self.img_dir_npz = img_dir_npz

        self.ids = json.load(open(ids_path))

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
        id_ = self.ids[i]

        # img input
        img_feat, img_pos_feat, _ = self._get_img_feat(id_)

        attn_masks = np.ones(img_feat.shape[0], dtype=np.int64)

        img_masks = None

        return (id_, img_feat, img_pos_feat, attn_masks, img_masks)

    def __len__(self):
        return len(self.ids)

    def _get_img_feat(self, id_):
        """get image features"""
        if ".jpg" in id_:
            filename = id_.replace(".jpg", ".npz")
        else:
            filename = id_ + ".npz"

        feat_path = os.path.join(self.img_dir_npz, filename)
        data = np.load(feat_path)

        np_att_feat = data['feat']
        np_pred_boxes = data['pred_boxes']
        np_scores = data['scores']
        np_pred_classes = data['pred_classes']
        np_width = data['width']
        np_height = data['height']

        att_feat = np.array(np_att_feat).astype(np.float32)
        att_feat = att_feat[:config.MAX_IMG_LEN, :]

        box_width = np_pred_boxes[:config.MAX_IMG_LEN, 2] - np_pred_boxes[:config.MAX_IMG_LEN, 0]
        box_height = np_pred_boxes[:config.MAX_IMG_LEN, 3] - np_pred_boxes[:config.MAX_IMG_LEN, 1]
        scaled_width = box_width / np_width
        scaled_height = box_height / np_height
        scaled_x = np_pred_boxes[:config.MAX_IMG_LEN, 0] / np_width
        scaled_y = np_pred_boxes[:config.MAX_IMG_LEN, 1] / np_height

        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]

        pred_boxes = np.concatenate((scaled_x, scaled_y,
                                     scaled_x + scaled_width,
                                     scaled_y + scaled_height,
                                     scaled_width, scaled_height,
                                     scaled_width * scaled_height), axis=1)
        pred_boxes = np.array(pred_boxes).astype(np.float32)

        scores = np.array(np_scores).astype(np.float32)
        scores = scores[:config.MAX_IMG_LEN]

        pred_classes = np.array(np_pred_classes).astype(np.float32)
        pred_classes = pred_classes[:config.MAX_IMG_LEN]

        img_bb = pred_boxes
        bb_num = att_feat.shape[0]

        return att_feat, img_bb, bb_num


def tdOne_collate(inputs):
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
    (ids, img_feats, img_pos_feats, attn_masks, img_masks) = map(list, unzip(inputs))

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    img_masks = pad_sequence_(img_masks, batch_first=True, padding_value=0)
    img_feat = _mask_img_feat(img_feat, img_masks)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    batch_size = img_feat.shape[0]
    out_size = attn_masks.size(1)

    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    batch = {'input_ids': None,
             'position_ids': None,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': None,
             'audio_feat': None,
             'audio_pos_ids': None,
             'txt_inputs': None,
             'txt_gts': None,
             'txt_masks': None,
             'txt_pos': None,
             'ids': ids,
             'img_masks': None,
             'audio_masks': None,
             'mask_types': None}
    return batch
