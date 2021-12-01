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
""" generator """
from collections import defaultdict
import numpy as np
from mindspore import Tensor


data_column = [
    'input_ids',
    'position_ids',
    'img_feat',
    'img_pos_feat',
    'audio_feat',
    'audio_pos_ids',
    'attention_mask',
    'gather_index',
    'txt_labels',
    'txt_mask',
    'txt_label_mask',
    'img_mask_tgt',
    'img_mask_tgt_mask',
    'img_masks',
    'mrc_label_target',
    'mrfr_feat_target',
    'audio_mask_tgt_mask',
    'audio_masks',
    'mafr_feat_target',
    'itm_target',
    'ma_neg_index',
    'ma_neg_sample',
    'mr_neg_index',
    'mr_neg_sample',
    'txt_gts',
    'txt_masks',
    'img_token_gts',
    'img_token_masks',
    'taskId'
]

data_column_audio = [
    'input_ids',
    'position_ids',
    'attention_mask',
    'mel_targets',
    'duration_targets',
    'speakers',
    'texts',
    'src_lens',
    'mel_lens',
    'audio_max_text_len',
    'audio_max_mel_len',
    'pitch_targets',
    'energy_targets'
]

# 'audio_mel_targets',
# 'audio_src_masks',
# 'audio_mel_masks',
# 'audio_duration_targets',

task2id = {
    'mlmThree': 0,
    'mrcThree': 1,
    'mrfrThree': 2,
    'mafrThree': 3,
    'macThree': 4,
    "itmThree": 5,
    'mrctThree': 6,
    "tdThree": 7,
    "idThree": 8,
    "adThree": 9,
    "ret": 10,
    "ftRet": 11
}


def get_batch_data(batch):
    """ get_batch_data """

    for key, value in batch.items():
        batch[key] = Tensor(value)

    input_ids = batch['input_ids']
    position_ids = batch['position_ids']

    img_feat = batch['img_feat']  # 176, 10,d 2048
    img_pos_feat = batch['img_pos_feat']  # 176, 10, 7

    audio_feat = batch['audio_feat']  # 176, 10, 512
    audio_pos_ids = batch['audio_pos_ids']  # 1, 10

    # attention_mask: 32 * 191
    attention_mask = batch['attention_mask']
    # gather_index 32 * 191
    gather_index = batch['gather_index']

    txt_labels = batch['txt_labels']
    txt_mask = batch['txt_mask']
    txt_label_mask = batch['txt_label_mask']

    img_mask_tgt = batch['img_mask_tgt']  # 176, 72
    img_mask_tgt_mask = batch['img_mask_tgt_mask']  # 352, 2
    img_masks = batch['img_masks']  # 176, 10
    mrc_label_target = batch['mrc_label_target']  # 352, 1

    audio_mask_tgt_mask = batch['audio_mask_tgt_mask']
    audio_masks = batch['audio_masks']

    mrfr_feat_target = batch['mrfr_feat_target']
    mafr_feat_target = batch['mafr_feat_target']

    itm_target = batch['itm_target']

    ma_neg_sample = batch['ma_neg_sample']
    mr_neg_index = batch['mr_neg_index']
    mr_neg_sample = batch['mr_neg_sample']

    txt_gts = batch['txt_gts']
    txt_masks = batch['txt_masks']

    img_token_gts = batch['img_token_gts']
    img_token_masks = batch['img_token_masks']

    taskId = batch['taskId']

    return (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
            audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
            txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
            mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
            txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
            txt_masks, img_token_gts, img_token_masks,
            taskId)


def get_batch_data_audio(batch):
    """ get_batch_data_audio """

    input_ids = batch['input_ids']
    position_ids = batch['position_ids']

    # attention_mask: 32 * 191
    attention_mask = batch['attention_mask']

    mel_targets = batch['mel_targets']
    duration_targets = batch['duration_targets']
    speakers = batch['speakers']
    texts = batch['texts']
    src_lens = batch['src_lens']
    mel_lens = batch['mel_lens']
    audio_max_text_len = batch['audio_max_text_len']
    audio_max_mel_len = batch['audio_max_mel_len']
    pitch_targets = batch['pitch_targets']
    energy_targets = batch['energy_targets']

    output = (input_ids, position_ids, attention_mask,
              mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
              audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets)

    return output


def get_batch_data_captioneval(batch):
    """ get_batch_data_captioneval """

    for key, val in batch.items():
        if isinstance(val, np.ndarray):
            if val.dtype == np.int64:
                batch[key] = val.astype(np.int32)

    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            batch[key] = Tensor(value)

    batch = defaultdict(lambda: None, batch)

    input_ids = batch.get('input_ids', None)
    position_ids = batch.get('position_ids', None)

    img_feat = batch['img_feat']  # self.bs, 10,d 2048
    img_pos_feat = batch['img_pos_feat']  # self.bs, 10, 7

    audio_feat = batch['audio_feat']  # self.bs, 10, 512
    audio_pos_ids = batch['audio_pos_ids']  # 1, 10

    # attention_mask: 32 * 191
    attention_mask = batch['attn_masks']
    # gather_index 32 * 191
    gather_index = batch['gather_index']

    txt_labels = batch['txt_labels']
    txt_mask = batch['txt_mask']
    txt_label_mask = batch['txt_label_mask']

    img_mask_tgt = batch['img_mask_tgt']  # self.bs, 72
    img_mask_tgt_mask = batch['img_mask_tgt_mask']  # self.bs*2, 2
    img_masks = batch['img_masks']  # self.bs, 10
    mrc_label_target = batch['label_targets']  # self.bs*2, 1

    audio_mask_tgt_mask = batch['audio_mask_tgt_mask']
    audio_masks = batch['audio_masks']

    mrfr_feat_target = batch.get('mrfr_feat_target', None)
    mafr_feat_target = batch.get('mafr_feat_target', None)

    itm_target = batch.get('targets', None)

    # ma_neg_index = batch.get('ma_neg_index', None)
    ma_neg_sample = batch.get('ma_neg_sample', None)
    mr_neg_index = batch.get('mr_neg_index', None)
    mr_neg_sample = batch.get('mr_neg_sample', None)

    txt_gts = batch.get('txt_gts', None)
    txt_masks = batch.get('txt_masks', None)

    img_token_gts = batch.get('img_token_gts', None)
    img_token_masks = batch.get('img_token_masks', None)

    taskID = None

    output = [input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
              audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
              txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
              mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
              txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
              txt_masks, img_token_gts, img_token_masks,
              taskID]

    for i in range(len(output)):
        if output[i] is None:
            output[i] = Tensor(np.ones((1, 1), dtype=np.float32))

    return tuple(output)
