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
Create dataset for training and evaluating
"""
from mindspore.dataset import GeneratorDataset

from data import MetaLoader, MetaLoaderAudio, data_column, data_column_audio
from tools.misc import set_random_seed
from pretrain_three_data import create_three_dataloaders


def get_input_data(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                   audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                   txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                   mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                   ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts, txt_masks,
                   img_token_gts, img_token_masks, taskId, rank, batch_size, device_num, full_batch):
    """ get_input_data """

    if full_batch:
        dis = batch_size
    else:
        dis = batch_size // device_num
    rank = int(rank)
    input_ids = input_ids[rank * dis: (rank + 1) * dis]
    position_ids = position_ids
    img_feat = img_feat[rank * dis: (rank + 1) * dis]
    img_pos_feat = img_pos_feat[rank * dis: (rank + 1) * dis]
    audio_feat = audio_feat[rank * dis: (rank + 1) * dis]
    audio_pos_ids = audio_pos_ids
    attention_mask = attention_mask[rank * dis: (rank + 1) * dis]
    gather_index = gather_index[rank * dis: (rank + 1) * dis]
    txt_labels = txt_labels[rank * dis: (rank + 1) * dis]
    txt_mask = txt_mask[rank * dis * 2: (rank + 1) * dis * 2]
    txt_label_mask = txt_label_mask[rank * dis * 2: (rank + 1) * dis * 2]
    img_mask_tgt = img_mask_tgt[rank * dis: (rank + 1) * dis]
    img_mask_tgt_mask = img_mask_tgt_mask[rank * dis * 2: (rank + 1) * dis * 2]
    img_masks = img_masks[rank * dis: (rank + 1) * dis]
    mrc_label_target = mrc_label_target[rank * dis * 2: (rank + 1) * dis * 2]
    mrfr_feat_target = mrfr_feat_target[rank * dis * 2: (rank + 1) * dis * 2]
    audio_mask_tgt_mask = audio_mask_tgt_mask[rank * dis * 2: (rank + 1) * dis * 2]
    audio_masks = audio_masks[rank * dis: (rank + 1) * dis]
    mafr_feat_target = mafr_feat_target[rank * dis * 2: (rank + 1) * dis * 2]
    itm_target = itm_target[rank * dis: (rank + 1) * dis]
    ma_neg_index = ma_neg_index[rank * dis * 2: (rank + 1) * dis * 2]
    ma_neg_sample = ma_neg_sample[rank * dis * 2: (rank + 1) * dis * 2]
    mr_neg_index = mr_neg_index[rank * dis * 2: (rank + 1) * dis * 2]
    mr_neg_sample = mr_neg_sample[rank * dis * 2: (rank + 1) * dis * 2]
    txt_gts = txt_gts[rank * dis: (rank + 1) * dis]
    txt_masks = txt_masks[rank * dis: (rank + 1) * dis]
    img_token_gts = img_token_gts[rank * dis: (rank + 1) * dis]
    img_token_masks = img_token_masks[rank * dis: (rank + 1) * dis]
    taskId = taskId
    return input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids, \
           attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt, \
           img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target, audio_mask_tgt_mask, \
           audio_masks, mafr_feat_target, itm_target, ma_neg_index, ma_neg_sample, mr_neg_index, \
           mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks, taskId


def create_dataset(opts, device_num=1, rank=0, column_name=None,
                   token_size=5120, full_batch=False, seq_length=28, mul_size=8, is_train=True, batch_size=-1,
                   print_time=True):
    """
    Create dataset

    Inputs:
        opts: config file which including dataset path
        device_num: total device number
        rank: current rank id
        column_name: the column name of the train file. Default is a list
        batch_size: batch size
        full_batch: whether do full batch operation.
        drop_remainder: whether drop remainder

    Returns:
        dataset_restore: the dataset for training
    """
    set_random_seed(opts.seed)
    if is_train:
        train_data_loaders = create_three_dataloaders(opts.ids_train_path, opts.train_datasets, is_train,
                                                      opts, device_num=device_num)
    else:
        train_data_loaders = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, is_train,
                                                      opts, device_num=device_num)
    # build data loaders
    if batch_size == -1:
        bs = token_size // seq_length
        bs = bs // mul_size * mul_size
    else:
        bs = batch_size
    per_batch = bs // device_num

    metaloader = MetaLoader(train_data_loaders, per_batch, task_num=len(train_data_loaders.keys()),
                            print_time=print_time)

    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=False)

    # If eod_reset enabled, another two inputs will be generated through input_ids
    return dataset


def create_dataset_metaloader(opts, device_num=1, rank=0, column_name=None,
                              token_size=5120, full_batch=False, seq_length=28, mul_size=8, is_train=True,
                              batch_size=-1):
    """
    Create dataset

    Inputs:
        opts: config file which including dataset path
        device_num: total device number
        rank: current rank id
        column_name: the column name of the train file. Default is a list
        batch_size: batch size
        full_batch: whether do full batch operation.
        drop_remainder: whether drop remainder

    Returns:
        dataset_restore: the dataset for training
    """
    set_random_seed(opts.seed)
    if is_train:
        train_data_loaders = create_three_dataloaders(opts.ids_train_path, opts.train_datasets, is_train,
                                                      opts, device_num=device_num)
    else:
        train_data_loaders = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, is_train,
                                                      opts, device_num=device_num)
    # build data loaders
    if batch_size == -1:
        bs = token_size // seq_length
        bs = bs // mul_size * mul_size
    else:
        bs = batch_size
    per_batch = bs // device_num
    metaloader = MetaLoader(train_data_loaders, per_batch, task_num=len(train_data_loaders.keys()))

    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=False)

    # If eod_reset enabled, another two inputs will be generated through input_ids
    return dataset, metaloader


def create_audio_dataset(opts, device_num=1, rank=0, column_name=None,
                         token_size=5120, full_batch=False, seq_length=30, mul_size=8, is_train=True):
    """
    Create dataset

    Inputs:
        opts: config file which including dataset path
        device_num: total device number
        rank: current rank id
        column_name: the column name of the train file. Default is a list
        batch_size: batch size
        full_batch: whether do full batch operation.
        drop_remainder: whether drop remainder

    Returns:
        dataset_restore: the dataset for training
    """
    set_random_seed(opts.seed)
    if is_train:
        train_data_loaders = create_three_dataloaders(opts.ids_train_path, opts.train_datasets, is_train,
                                                      opts, device_num=device_num)
    else:
        train_data_loaders = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, is_train,
                                                      opts, device_num=device_num)
    # build data loaders
    bs = token_size // seq_length
    bs = bs // mul_size * mul_size
    per_batch = bs // device_num
    metaloader = MetaLoaderAudio(train_data_loaders, per_batch, task_num=len(train_data_loaders.keys()))
    dataset = GeneratorDataset(metaloader, column_names=data_column_audio, shuffle=False)

    # If eod_reset enabled, another two inputs will be generated through input_ids
    return dataset
