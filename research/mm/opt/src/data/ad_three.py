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
""" ad_three """

import random
import os
import json
from toolz.sandbox import unzip
from fastspeech2_ms.text import text_to_sequence
import numpy as np
from .data import get_gather_index, pad_tensors
from .data_three import (DetectFeatTxtTokAudioFeatDataset, TxtTokDataset)
from .mlm_three import random_word_with_all, random_word_fix
from .mrm_three import _get_img_mask, _get_img_mask_with_all

from .utils import pad_sequence, pad_sequence_


def _mask_img_feat(img_feat, img_masks):
    """ _mask_img_feat """
    if img_masks is None or img_masks[0] is None:
        return img_feat
    # img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_masks_ext = np.broadcast_to(np.expand_dims(img_masks, -1), img_feat.shape)
    img_feat[img_masks_ext] = 0
    return img_feat


class AdThreeDataset(DetectFeatTxtTokAudioFeatDataset):
    """ AdThreeDataset """

    def __init__(self, ids_path, txt_db, img_db, audio_mel_path, mask_type=-1, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, use_video=use_video)
        self.mask_type = mask_type
        self.use_video = use_video
        self.audio_mel_path = audio_mel_path
        self.use_mask_fix = use_mask_fix

    def get_mel(self, id_):
        """ get_mel """

        npz_path = os.path.join(self.audio_mel_path, id_ + ".npz")
        feat = np.load(npz_path)['feat'][:, :, :1000]
        feat = feat[0].transpose(1, 0)

        return feat

    def mask_text(self, input_ids, mask_prob):
        """ mask_text """

        if self.use_mask_fix:
            input_ids, txt_labels = random_word_fix(input_ids,
                                                    self.txt_db.v_range,
                                                    self.txt_db.mask)
        else:
            input_ids, txt_labels = random_word_with_all(input_ids,
                                                         self.txt_db.v_range,
                                                         self.txt_db.mask,
                                                         mask_prob)

        # add mask
        input_ids, txt_labels = random_word_with_all(input_ids,
                                                     self.txt_db.v_range,
                                                     self.txt_db.mask, mask_prob)
        # add cls and sep
        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        txt_labels = np.array([-1] + txt_labels + [-1])
        return input_ids, txt_labels

    def mask_none(self, input_ids):
        """ mask_none """

        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        return input_ids, None

    def mask_txt_img_with_type(self, input_ids, num_bb):
        """ mask_txt_img_with_type """

        mask_type = self.mask_type
        if self.mask_type == 0:
            txt_labels = None
            audio_mask = _get_img_mask_with_all(1.1, num_bb)
        elif self.mask_type == 1:
            input_ids, txt_labels = self.mask_text(input_ids, 1.1)
            audio_mask = None
        elif self.mask_type == 2:
            input_ids, txt_labels = self.mask_none(input_ids)
            audio_mask = None
        else:
            raise Exception("Error Mask Type {}".format(self.mask_type))

        return input_ids, txt_labels, audio_mask, mask_type

    def mask_txt_img(self, input_ids, num_bb):
        """ mask_txt_img """

        prob = random.random()
        if prob < 0.3:  # make all text
            input_ids, txt_labels = self.mask_text(input_ids, 1.1)
            img_mask = _get_img_mask(0.15, num_bb)
            mask_type = 0
        elif prob < 0.6:  # make all audio
            input_ids, txt_labels = self.mask_text(input_ids, 0.15)
            img_mask = _get_img_mask(1.1, num_bb)
            mask_type = 1
        else:  # make text and audio
            input_ids, txt_labels = self.mask_text(input_ids, 0.15)
            img_mask = _get_img_mask(0.15, num_bb)
            mask_type = 2

        return input_ids, txt_labels, img_mask, mask_type

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

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

        # mask
        if self.mask_type >= 0:
            input_ids, txt_labels, img_mask, mask_type = self.mask_txt_img_with_type(example['input_ids'], num_bb)
        else:
            input_ids, txt_labels, img_mask, mask_type = self.mask_txt_img(example['input_ids'], num_bb)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0], dtype=np.int32)

        mel_feat = self.get_mel(ids)
        mel_mask = np.ones(mel_feat.shape[0], dtype=np.int32)

        mel_every_duration = round(mel_feat.shape[0] // attn_masks.shape[0])
        mel_duration = [mel_every_duration for _ in range(attn_masks.shape[0])]

        return (ids, input_ids, txt_labels, mask_type,
                img_feat, img_pos_feat, img_mask, attn_masks,
                mel_feat, mel_mask, mel_duration)


def adThree_collate(inputs):
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
    (ids, input_ids, txt_labels, mask_type,
     img_feats, img_pos_feats, img_masks, attn_masks,
     mel_feats, mel_masks, mel_durations) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence_(txt_labels, batch_first=True, padding_value=-1)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    img_masks = pad_sequence_(img_masks, batch_first=True, padding_value=0)
    img_feat = _mask_img_feat(img_feat, img_masks)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.shape
    out_size = attn_masks.shape[1]

    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    # mels
    num_mels = [f.shape[0] for f in mel_feats]
    mel_feats = pad_tensors(mel_feats, num_mels)
    mel_masks = pad_sequence_(mel_masks, batch_first=True, padding_value=0)
    duration_targets = pad_sequence(mel_durations, batch_first=True, padding_value=0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels,
             'audio_feat': None,
             'audio_pos_ids': None,
             'ids': ids,
             'img_masks': img_masks,
             'audio_masks': None,
             'audio_mel_targets': mel_feats,
             'audio_src_masks': attn_masks,
             'audio_mel_masks': mel_masks,
             'audio_duration_targets': duration_targets,
             'mask_type': mask_type}
    return batch


class AdTextV3Dataset(TxtTokDataset):
    """ AdTextV3Dataset """

    def __init__(self, ids_path, txt_db, audio_mel_path, preprocess_config, mask_type=-1):
        super().__init__(ids_path, txt_db)
        self.mask_type = mask_type
        self.audio_mel_path = audio_mel_path

        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            os.path.join(self.audio_mel_path, "all.txt")
        )
        with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                )
        ) as f:
            self.speaker_map = json.load(f)

    def process_meta(self, filename):
        """ process_meta """

        # SSB03390295|SSB0339|{n i3 m en5 q ve4 sh iii2 y iou3 y i2 g e4 g ong4 t ong1 sp d ian3}
        # |ni3 men5 que4 shi2 you3 yi2 ge4 gong4 tong1 dian3
        with open(filename, "r", encoding="utf-8") as f:
            name = {}
            speaker = {}
            text = {}
            raw_text = {}
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")

                new_id = "aishell3/{}-{}".format(s, n)

                name[new_id] = n
                speaker[new_id] = s
                text[new_id] = t
                raw_text[new_id] = r

            return name, speaker, text, raw_text

    def get_feat(self, id_, name):
        """ get_feat """

        id1, id2 = id_.split("/")[-1].split("-")
        new_name = "{}-{}-{}.npy".format(id1, name, id2)
        npy_path = os.path.join(self.audio_mel_path, name, new_name)
        feat = np.load(npy_path)
        return feat

    def mask_text(self, input_ids):
        """ mask_text """

        # add mask
        input_ids, txt_labels = random_word_fix(input_ids,
                                                self.txt_db.v_range,
                                                self.txt_db.mask)

        # add cls and sep
        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        txt_labels = np.array([-1] + txt_labels + [-1])
        return input_ids, txt_labels

    def mask_none(self, input_ids):
        """ mask_none """

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

        # mask
        if self.mask_type >= 0:
            input_ids, txt_labels = self.mask_none(example['input_ids'])
        else:
            input_ids, txt_labels = self.mask_text(example['input_ids'])

        attn_masks = np.ones(len(input_ids), dtype=np.int64)

        mel_feat = self.get_feat(ids, "mel")
        mel_mask = np.ones(mel_feat.shape[0], dtype=np.int64)

        mel_duration = self.get_feat(ids, "duration")

        energy_feat = self.get_feat(ids, "energy")
        pitch_feat = self.get_feat(ids, "pitch")

        speaker = self.speaker[ids]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[ids], self.cleaners))
        raw_text = self.raw_text[ids]

        return (ids, input_ids, txt_labels,
                attn_masks,
                mel_feat, mel_mask, mel_duration,
                energy_feat, pitch_feat,
                speaker_id, phone, raw_text)


def adTextV3_collate(inputs):
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
    (ids, input_ids, txt_labels,
     attn_masks,
     mel_feats, mel_masks, mel_durations, energy_feats, pitch_feats,
     speaker_ids, phones, raw_texts) = map(list, unzip(inputs))

    # text batches
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence_(txt_labels, batch_first=True, padding_value=-1, max_lens=30)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    out_size = attn_masks.shape[1]
    bs, _ = input_ids.shape

    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0)
    gather_index = np.broadcast_to(gather_index, (bs, out_size))

    # mels
    num_mels = [f.shape[0] for f in mel_feats]
    mel_feats = pad_tensors(mel_feats, num_mels, max_len=1298)
    mel_masks = pad_sequence_(mel_masks, batch_first=True, padding_value=0, max_lens=1298)

    duration_targets = pad_sequence(mel_durations, batch_first=True, padding_value=0, max_lens=89)
    energy_targets = pad_sequence(energy_feats, batch_first=True, padding_value=0, max_lens=89)
    pitch_targets = pad_sequence(pitch_feats, batch_first=True, padding_value=0, max_lens=89)

    speaker_ids = np.array(speaker_ids)
    texts = pad_sequence(phones, batch_first=True, padding_value=0, max_lens=89)

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array(num_mels)

    audio_max_text_len = 89
    audio_max_mel_len = 1289

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels,
             'ids': ids,
             'audio_mel_targets': mel_feats,
             'audio_src_masks': attn_masks,
             'audio_mel_masks': mel_masks,
             'audio_duration_targets': duration_targets,
             'audio_energy_targets': energy_targets,
             'audio_pitch_targets': pitch_targets,
             'audio_speakers': speaker_ids,
             'audio_texts': texts,
             'audio_text_lens': text_lens,
             'audio_mel_lens': mel_lens,
             'audio_max_text_len': audio_max_text_len,
             'audio_max_mel_len': audio_max_mel_len,
             'raw_texts': raw_texts}
    return batch
