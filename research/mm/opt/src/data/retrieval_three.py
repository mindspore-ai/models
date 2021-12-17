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
"""retrieval three"""
import os
from toolz.sandbox import unzip
import numpy as np

from .utils import pad_sequence
from .data import pad_tensors_pos
from .data_three import (get_gather_index_three,
                         DetectFeatTxtTokAudioFeatDataset, get_ids_three,
                         TxtTokThreeLmdb, DetectFeatThreeLmdb, AudioFeatThreeLmdb)
from .data import pad_tensors, get_gather_index


class TxtAudiotoImgEvalDataset(DetectFeatTxtTokAudioFeatDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, ids_path, txt_db, img_db, audio_db, neg_sample_p=0.5, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)
        assert isinstance(audio_db, AudioFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        self.path_lens = self.path_lens.split('.')[0] + '_itm' + str(rank_id) + '.json'

        print("itm ids {}".format(self.total_len))

        self.neg_sample_p = neg_sample_p

    def __len__(self):
        return self.total_len * self.total_len

    def __getitem__(self, i):
        i1 = i // self.total_len
        i2 = i % self.total_len
        example1 = super().__getitem__(i1)
        example2 = super().__getitem__(i2)
        # labels and negative images should be sampled every epoch

        # text input
        input_ids = example1['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:8]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image
        img_id = example2['img_fname']
        img_feat, img_pos_feat, _ = self._get_img_feat(img_id)

        # audio
        audio_id = example1['audio_fname']
        audio_feat, _ = self._get_audio_feat(audio_id)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        return input_ids, img_feat, img_pos_feat, attn_masks, audio_feat


class TxttoImgEvalDataset(DetectFeatTxtTokAudioFeatDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, ids_path, txt_db, img_db, audio_db, neg_sample_p=0.5, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)
        assert isinstance(audio_db, AudioFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        self.path_lens = self.path_lens.split('.')[0] + '_itm' + str(rank_id) + '.json'

        print("itm ids {}".format(self.total_len))

        self.neg_sample_p = neg_sample_p

    def __len__(self):
        return self.total_len * self.total_len

    def __getitem__(self, i):
        i1 = i // self.total_len
        i2 = i % self.total_len
        example1 = super().__getitem__(i1)
        example2 = super().__getitem__(i2)
        # labels and negative images should be sampled every epoch

        # text input
        input_ids = example1['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:8]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image
        img_id = example2['img_fname']
        img_feat, img_pos_feat, _ = self._get_img_feat(img_id)

        # audio
        audio_feat = None

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0], dtype=np.int64)

        return input_ids, img_feat, img_pos_feat, attn_masks, audio_feat


class AudiotoImgEvalDataset(DetectFeatTxtTokAudioFeatDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, ids_path, txt_db, img_db, audio_db, neg_sample_p=0.5, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)
        assert isinstance(audio_db, AudioFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        self.path_lens = self.path_lens.split('.')[0] + '_itm' + str(rank_id) + '.json'

        print("itm ids {}".format(self.total_len))

        self.neg_sample_p = neg_sample_p

    def __len__(self):
        return self.total_len * self.total_len

    def __getitem__(self, i):
        i1 = i // self.total_len
        i2 = i % self.total_len
        example1 = super().__getitem__(i1)
        example2 = super().__getitem__(i2)
        # labels and negative images should be sampled every epoch

        # text input
        input_ids = [1000]
        if self.use_mask_fix:
            input_ids = input_ids[:8]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image
        img_id = example2['img_fname']
        img_feat, img_pos_feat, _ = self._get_img_feat(img_id)

        # audio
        audio_id = example1['audio_fname']
        audio_feat, _ = self._get_audio_feat(audio_id)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        return input_ids, img_feat, img_pos_feat, attn_masks, audio_feat


class TxttoAudioEvalDataset(DetectFeatTxtTokAudioFeatDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, ids_path, txt_db, img_db, audio_db, neg_sample_p=0.5, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)
        assert isinstance(audio_db, AudioFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        self.path_lens = self.path_lens.split('.')[0] + '_itm' + str(rank_id) + '.json'

        print("itm ids {}".format(self.total_len))

        self.neg_sample_p = neg_sample_p

    def __len__(self):
        return self.total_len * self.total_len

    def __getitem__(self, i):
        i1 = i // self.total_len
        i2 = i % self.total_len
        example1 = super().__getitem__(i1)
        example2 = super().__getitem__(i2)
        # labels and negative images should be sampled every epoch

        # text input
        input_ids = example1['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:8]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image
        img_feat = None
        img_pos_feat = None
        # audio
        audio_id = example2['audio_fname']
        audio_feat, _ = self._get_audio_feat(audio_id)

        attn_masks = np.ones(len(input_ids) + audio_feat.shape[0], dtype=np.int64)

        return input_ids, img_feat, img_pos_feat, attn_masks, audio_feat


class TxtImgtoAudioEvalDataset(DetectFeatTxtTokAudioFeatDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, ids_path, txt_db, img_db, audio_db, neg_sample_p=0.5, use_video=False, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db, use_video)
        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)
        assert isinstance(audio_db, AudioFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_video = use_video
        self.use_mask_fix = use_mask_fix

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        self.path_lens = self.path_lens.split('.')[0] + '_itm' + str(rank_id) + '.json'

        print("itm ids {}".format(self.total_len))

        self.neg_sample_p = neg_sample_p

    def __len__(self):
        return self.total_len * self.total_len

    def __getitem__(self, i):
        i1 = i // self.total_len
        i2 = i % self.total_len
        example1 = super().__getitem__(i1)
        example2 = super().__getitem__(i2)
        # labels and negative images should be sampled every epoch

        # text input
        input_ids = example1['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:8]
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image
        img_id = example1['img_fname']
        img_feat, img_pos_feat, _ = self._get_img_feat(img_id)

        # audio
        audio_id = example2['audio_fname']
        audio_feat, _ = self._get_audio_feat(audio_id)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        return input_ids, img_feat, img_pos_feat, attn_masks, audio_feat


def itmMatchingThree_collate(inputs):
    """Text-Image-Audio Matching dataset collate function"""
    (input_ids, img_feats, img_pos_feats, attn_masks, audio_feats
     ) = map(list, unzip(inputs))
    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

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

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids}
    return batch


def itmMatchingTxtImg_collate(inputs):
    """Text-Image Matching dataset collate function"""
    (input_ids, img_feats, img_pos_feats, attn_masks, _
     ) = map(list, unzip(inputs))
    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=30)

    bs, max_tl = input_ids.shape
    out_size = attn_masks.shape[1]

    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'audio_feat': None,
             'audio_pos_ids': None}
    return batch


def itmMatchingTxtAudio_collate(inputs):
    """Text-Audio Matching dataset collate function"""
    (input_ids, _, _, attn_masks, audio_feats
     ) = map(list, unzip(inputs))
    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    # audio batches
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=30)

    bs, max_tl = input_ids.shape
    out_size = attn_masks.shape[1]

    gather_index = get_gather_index(txt_lens, num_aus, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': None,
             'img_pos_feat': None,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids}
    return batch
