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
"""loader"""
import random
import time
from collections import defaultdict
from multiprocessing import Process
import numpy as np
from .data_loader import DataLoader

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
    "ftRet": 11,
    "ftCap": 12
}

task2id_open = {
    'mlmThree_open': 0,
    'mrcThree_open': 1,
    'mrfrThree_open': 2,
    'mafrThree_open': 3,
    'macThree_open': 4,
    "itmThree_open": 5,
    'mrctThree_open': 6,
    "tdThree_open": 7,
    "idThree_open": 8,
    "adThree_open": 9
}


class MetaLoader():
    """ wraps multiple data loaders """

    def __init__(self, loaders, batch_size=176, accum_steps=1, task_num=9, print_time=True):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter2 = {}
        self.sampling_pools = []
        self.loaders = loaders
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2iter2[n] = iter(l)
            self.sampling_pools.extend([n] * r)
        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.accum_steps = accum_steps
        self.bs = batch_size
        self.step_cnt = 0
        self.flag = "iter1"
        self.iter1_init_cnt = 0
        self.iter2_init_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        random.seed(1)
        self.all_ids = []
        self.print_time = print_time

    def init_iter(self, init_cnt):
        while init_cnt < self.task_num:
            local_task = self.sampling_pools[init_cnt]
            iter_tmp = iter(self.name2loader[local_task])
            if self.flag == 'iter1':
                self.name2iter2[local_task] = iter_tmp
            else:
                self.name2iter[local_task] = iter_tmp
            init_cnt += 1

    def return_ids(self):
        return self.all_ids

    def get_batch_params(self, batch):
        """ get_batch_params """

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

        ma_neg_index = batch.get('ma_neg_index', None)
        ma_neg_sample = batch.get('ma_neg_sample', None)
        mr_neg_index = batch.get('mr_neg_index', None)
        mr_neg_sample = batch.get('mr_neg_sample', None)

        txt_gts = batch.get('txt_gts', None)
        txt_masks = batch.get('txt_masks', None)

        img_token_gts = batch.get('img_token_gts', None)
        img_token_masks = batch.get('img_token_masks', None)

        return (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                txt_masks, img_token_gts, img_token_masks)

    def get_batch_check(self, batch, input_ids, position_ids, audio_feat,
                        audio_pos_ids, attention_mask, txt_labels, txt_mask,
                        txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                        mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                        ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                        txt_masks, img_token_gts, img_token_masks):
        """ get_batch_check """

        ids = batch.get('ids', None)
        if ids is not None:
            self.all_ids = self.all_ids + ids
        self.bs = attention_mask.shape[0]
        # text
        if input_ids is None:
            input_ids = np.zeros((self.bs, 30)).astype(np.int32)
        if position_ids is None:
            position_ids = np.zeros((1, 30)).astype(np.int32)
        if txt_labels is None:
            txt_labels = np.zeros((self.bs, 30)).astype(np.int32)
        if txt_mask is None:
            txt_mask = np.zeros((self.bs * 2, 5)).astype(np.int32)
        if txt_label_mask is None:
            txt_label_mask = np.zeros(self.bs * 2).astype(np.int32)

        # image
        if img_mask_tgt is None:
            img_mask_tgt = np.zeros((self.bs, 90)).astype(np.bool_)
        if img_mask_tgt_mask is None:
            img_mask_tgt_mask = np.zeros((self.bs * 2, 5)).astype(np.int32)
        if img_masks is None:
            img_masks = np.zeros((self.bs, 30)).astype(np.bool_)
        if mrc_label_target is None:
            mrc_label_target = np.zeros((self.bs * 2, 1)).astype(np.float32)

        # audio
        if audio_feat is None:
            audio_feat = np.zeros((self.bs, 30, 1024)).astype(np.float32)  # 用attention_mask.shape[0]替换了self.bs
        if audio_pos_ids is None:
            audio_pos_ids = np.zeros((1, 30)).astype(np.int32)

        if mrfr_feat_target is None:
            mrfr_feat_target = np.zeros((self.bs * 2, 2048)).astype(np.float32)

        if audio_mask_tgt_mask is None:
            audio_mask_tgt_mask = np.zeros((self.bs * 2, 5)).astype(np.int32)
        if audio_masks is None:
            audio_masks = np.zeros((self.bs, 30)).astype(np.bool_)

        if mafr_feat_target is None:
            mafr_feat_target = np.zeros((self.bs * 2, 1024)).astype(np.float32)

        if itm_target is None:
            itm_target = np.zeros((self.bs,)).astype(np.int32)
        if ma_neg_index is None:
            ma_neg_index = np.zeros((self.bs * 2, 1)).astype(np.int32)
        if ma_neg_sample is None:
            ma_neg_sample = np.zeros((self.bs * 2, 30, 1024)).astype(np.float32)
        if mr_neg_index is None:
            mr_neg_index = np.zeros((self.bs * 2, 1)).astype(np.int32)
        if mr_neg_sample is None:
            mr_neg_sample = np.zeros((self.bs * 2, 30, 2048)).astype(np.float32)
        if txt_gts is None:
            txt_gts = np.zeros((self.bs, 90)).astype(np.int32)
        if txt_masks is None:
            txt_masks = np.ones((self.bs, 90)).astype(np.float32)

        if img_token_gts is None:
            img_token_gts = np.zeros((self.bs, 64)).astype(np.int32)
        if img_token_masks is None:
            img_token_masks = np.ones((self.bs, 64)).astype(np.float32)

        return (input_ids, position_ids, audio_feat,
                audio_pos_ids, attention_mask, txt_labels, txt_mask,
                txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                txt_masks, img_token_gts, img_token_masks)

    def get_batch(self, batch, task):
        """ get_batch """

        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
         audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks) = self.get_batch_params(batch)

        (input_ids, position_ids, audio_feat,
         audio_pos_ids, attention_mask, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks) = self.get_batch_check(batch, input_ids, position_ids, audio_feat,
                                                                           audio_pos_ids, attention_mask, txt_labels,
                                                                           txt_mask,
                                                                           txt_label_mask, img_mask_tgt,
                                                                           img_mask_tgt_mask, img_masks,
                                                                           mrc_label_target,
                                                                           mrfr_feat_target, audio_mask_tgt_mask,
                                                                           audio_masks, mafr_feat_target, itm_target,
                                                                           ma_neg_index, ma_neg_sample, mr_neg_index,
                                                                           mr_neg_sample, txt_gts,
                                                                           txt_masks, img_token_gts, img_token_masks)

        if self.print_time:
            print("txt: {} img:{} audio:{}".format(input_ids.shape, img_feat.shape, audio_feat.shape))
        taskId = np.array([task2id[task]]).astype(np.int32)
        txt_masks = txt_masks.astype(np.float32)

        output = (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                  audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                  txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                  mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                  txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                  txt_masks, img_token_gts, img_token_masks, taskId)

        return output

    def __getitem__(self, index):
        start_time = time.time()
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        self.step_cnt += 1
        task_index = self.task_index_list[self.step_cnt - 1]
        local_task = self.sampling_pools[task_index]

        if self.flag == "iter1":
            iter_ = self.name2iter[local_task]
        else:
            iter_ = self.name2iter2[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            print("Epoch End.")
            if self.task_num == 1:
                self.flag = "iter2"
                self.init_iter(0)
                iter_ = self.name2iter[local_task]
                batch = next(iter_)
                self.flag = "iter1"
            else:
                if self.flag == "iter1":
                    self.flag = "iter2"
                else:
                    self.flag = "iter1"

                if self.flag == "iter1":
                    self.iter1_init_cnt = 0
                    iter_ = self.name2iter[local_task]
                    batch = next(iter_)
                    p = Process(target=self.init_iter, args=(self.iter2_init_cnt,))
                    p.start()

                else:

                    self.iter2_init_cnt = 0
                    iter_ = self.name2iter2[local_task]
                    batch = next(iter_)
                    p = Process(target=self.init_iter, args=(self.iter1_init_cnt,))
                    p.start()

        task = name.split('_')[0]
        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)

        if self.print_time:
            print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        return output

    def __len__(self):
        # return 256*64
        return 40


class MetaLoaderAudio():
    """ wraps multiple data loaders """

    def __init__(self, loaders, batch_size=176, accum_steps=1, task_num=9):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter2 = {}
        self.sampling_pools = []
        self.loaders = loaders
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2iter2[n] = iter(l)
            self.sampling_pools.extend([n] * r)
        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.accum_steps = accum_steps
        self.bs = batch_size
        self.step_cnt = 0
        self.flag = "iter1"
        self.iter1_init_cnt = 0
        self.iter2_init_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        random.seed(1)
        self.all_ids = []

    def init_iter(self, init_cnt):
        while init_cnt < self.task_num:
            local_task = self.sampling_pools[init_cnt]
            iter_tmp = iter(self.name2loader[local_task])
            if self.flag == 'iter1':
                self.name2iter2[local_task] = iter_tmp
            else:
                self.name2iter[local_task] = iter_tmp
            init_cnt += 1

    def return_ids(self):
        return self.all_ids

    def __getitem__(self, index):
        start_time = time.time()
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        self.step_cnt += 1
        task_index = self.task_index_list[self.step_cnt - 1]
        local_task = self.sampling_pools[task_index]

        if self.flag == "iter1":
            iter_ = self.name2iter[local_task]
        else:
            iter_ = self.name2iter2[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            print("Epoch End.")
            if self.flag == "iter1":
                self.flag = "iter2"
            else:
                self.flag = "iter1"
            if self.flag == "iter1":
                self.iter1_init_cnt = 0
                iter_ = self.name2iter[local_task]
                batch = next(iter_)
                p = Process(target=self.init_iter, args=(self.iter2_init_cnt,))
                p.start()
            else:
                self.iter2_init_cnt = 0
                iter_ = self.name2iter2[local_task]
                batch = next(iter_)
                p = Process(target=self.init_iter, args=(self.iter1_init_cnt,))
                p.start()
        task = name.split('_')[0]
        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        batch = defaultdict(lambda: None, batch)


        input_ids = batch.get('input_ids', None)
        position_ids = batch.get('position_ids', None)

        # attention_mask: 32 * 191
        attention_mask = batch['attn_masks']

        mel_targets = batch['audio_mel_targets']
        duration_targets = batch['audio_duration_targets']
        speakers = batch['audio_speakers']
        texts = batch['audio_texts']
        src_lens = batch['audio_text_lens']
        mel_lens = batch['audio_mel_lens']
        audio_max_text_len = batch['audio_max_text_len']
        audio_max_mel_len = batch['audio_max_mel_len']
        pitch_targets = batch['audio_pitch_targets']
        energy_targets = batch['audio_energy_targets']

        ids = batch.get('ids', None)
        if ids is not None:
            self.all_ids = self.all_ids + ids

        output = (input_ids, position_ids, attention_mask,
                  mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
                  audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets)

        print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        return output

    def __len__(self):
        # return 256*64
        return 40
