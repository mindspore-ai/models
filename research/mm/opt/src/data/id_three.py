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
import os
import numpy as np
from config import config
from toolz.sandbox import unzip
from .data import pad_tensors_pos, pad_tensors
from .data_three import DetectFeatTxtTokAudioFeatDataset, get_gather_index_three
from .mlm_three import random_word_with_all
from .mrm_three import _get_img_mask_with_all
from .mam_three import _get_audio_mask


from .utils import pad_sequence, pad_sequence_


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


class IdThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """
    IdThreeDataset
    """
    def __init__(self, ids_path, txt_db, img_db, audio_db, img_token_path, mask_type=-1, data_type=0):
        super().__init__(ids_path, txt_db, img_db, audio_db)
        self.img_token_path = img_token_path
        self.mask_type = mask_type
        self.data_type = data_type
        self.mask_prob = 1.1

    def get_img_token(self, id_):
        """
        get_img_token
        """
        if self.data_type == 1:
            npz_path = os.path.join(self.img_token_path, id_.replace("_", "/") + ".npz")
        elif self.data_type == 2:
            if ".jpg" in id_:
                num = id_.split(".jpg")[-1]
                npz_path = os.path.join(self.img_token_path, id_.replace(".jpg" + num, ".jpg.npz"))
                if not os.path.exists(npz_path):
                    npz_path = os.path.join(self.img_token_path, id_.replace(".jpg" + num, ".npz"))
            else:
                npz_path = os.path.join(self.img_token_path, id_ + ".npz")
        else:
            npz_path = os.path.join(self.img_token_path, id_ + ".npz")
        feat = np.load(npz_path)['feat'].reshape(-1) + 1
        img_tokens = feat.tolist()[:config.MAX_IMG_GTS_LEN]

        img_token_inputs = np.array([0] + img_tokens)
        img_token_gts = np.array(img_tokens + [0])
        img_token_masks = np.ones(len(img_token_inputs))

        return img_token_inputs, img_token_gts, img_token_masks

    def mask_txt_img_audio(self, input_ids, num_audio):
        """mask_txt_img_audio """

        prob = random.random()
        if prob < 0.3:  # make all text
            input_ids, txt_labels = self.mask_text(input_ids, 1.1)
            audio_mask = _get_audio_mask(0.15, num_audio)
            mask_type = 0
        elif prob < 0.6:  # make all audio
            input_ids, txt_labels = self.mask_text(input_ids, 0.15)
            audio_mask = _get_audio_mask(1.1, num_audio)
            mask_type = 1
        else:  # make text and audio
            input_ids, txt_labels = self.mask_text(input_ids, 0.15)
            audio_mask = _get_audio_mask(0.15, num_audio)
            mask_type = 2

        return input_ids, txt_labels, audio_mask, mask_type

    def mask_txt_img_audio_with_type(self, input_ids, num_audio):
        """mask_txt_img_audio_with_type"""
        mask_type = self.mask_type
        if self.mask_type == 0:
            txt_labels = None
            audio_mask = _get_audio_mask(1.1, num_audio)
        elif self.mask_type == 1:
            input_ids, txt_labels = self.mask_text(input_ids, 1.1)
            audio_mask = None
        elif self.mask_type == 2:
            input_ids, txt_labels = self.mask_none(input_ids)
            audio_mask = None
        else:
            raise Exception("Error Mask Type {}".format(self.mask_type))

        return input_ids, txt_labels, audio_mask, mask_type

    def mask_text(self, input_ids, mask_prob):
        """mask_text"""
        # add mask
        input_ids, txt_labels = random_word_with_all(input_ids,
                                                     self.txt_db.v_range,
                                                     self.txt_db.mask,
                                                     mask_prob)
        # add cls and sep
        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        txt_labels = np.array([-1] + txt_labels + [-1])
        return input_ids, txt_labels

    def mask_none(self, input_ids):
        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        return input_ids, None

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

        # image input features
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        img_mask = _get_img_mask_with_all(self.mask_prob, num_bb)

        audio_feat, num_audio = self._get_audio_feat(example['audio_fname'])

        # mask
        if self.mask_type >= 0:
            input_ids, txt_labels, audio_mask, mask_type = self.mask_txt_img_audio_with_type(
                example['input_ids'][:config.MAX_TEXT_LEN], num_audio)
        else:
            input_ids, txt_labels, audio_mask, mask_type = self.mask_txt_img_audio(
                example['input_ids'][:config.MAX_TEXT_LEN], num_audio)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int32)

        # img token
        img_token_inputs, img_token_gts, img_token_masks = self.get_img_token(ids)

        return (ids, input_ids, attn_masks, txt_labels, img_feat, img_pos_feat, img_mask, audio_feat,
                audio_mask, mask_type,
                img_token_inputs, img_token_gts, img_token_masks)


def idThree_collate(inputs):
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
    (ids, input_ids, attn_masks, txt_labels, img_feats, img_pos_feats, img_masks, audio_feats,
     audio_masks, mask_types,
     img_token_inputs, img_token_gts, img_token_masks) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence_(txt_labels, batch_first=True, padding_value=-1)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)

    img_feat = _mask_img_feat(img_feat, img_masks)

    # audio batches
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats, num_aus)
    audio_pos_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                             ), 0)
    audio_masks = pad_sequence_(audio_masks, batch_first=True, padding_value=0)
    audio_feat = _mask_audio_feat(audio_feat, audio_masks)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=90)

    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)

    # img tokens
    img_token_inputs = pad_sequence(img_token_inputs, batch_first=True, padding_value=0, max_lens=64)
    img_token_gts = pad_sequence(img_token_gts, batch_first=True, padding_value=0, max_lens=64)
    img_token_masks = pad_sequence(img_token_masks, batch_first=True, padding_value=0, max_lens=64).astype(np.float32)
    img_token_pos = np.expand_dims(np.arange(0, img_token_inputs.shape[1], dtype=np.int64), 0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': None,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids,
             'ids': ids,
             'img_masks': None,
             'audio_masks': audio_masks,
             'mask_types': mask_types,
             'img_token_inputs': img_token_inputs,
             'img_token_gts': img_token_gts,
             'img_token_masks': img_token_masks,
             'img_token_pos': img_token_pos}
    return batch
