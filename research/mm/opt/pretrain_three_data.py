# Copyright 2020 Huawei Technologies Co., Ltd
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
UNITER pre-training
"""

from src.data import (TokenBucketPathSampler, TokenBucketPathSamplerForItm, BatchSampler, TxtTokThreeLmdb,
                      DetectFeatThreeLmdb, AudioFeatThreeLmdb,
                      MlmThreeDataset, MrcThreeDataset, MrfrThreeDataset, ItmThreeDataset, MafrThreeDataset,
                      TdThreeDataset, tdThree_collate,
                      TdOneDataset, tdOne_collate,
                      mlmThree_collate, mrcThree_collate, mrfrThree_collate, itmThree_collate, mafrThree_collate,
                      IdThreeDataset, idThree_collate)
from src.data import (itmMatchingThree_collate, TxttoImgEvalDataset, itmMatchingTxtImg_collate,
                      itmMatchingTxtAudio_collate,
                      TxtImgtoAudioEvalDataset, AudiotoImgEvalDataset, TxttoAudioEvalDataset, TxtAudiotoImgEvalDataset)
from src.data import CaptionDataset, caption_collate
from src.data import DataLoader
from src.data.retrieval_ft import ItmFlickrRankDataset, itm_rank_collate
from src.tools.const import BUCKET_SIZE
from src.tools.logger import LOGGER


def build_dataloader_ms(dataset, collate_fn, is_train, opts, device_num):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketPathSampler(dataset.path_lens, bucket_size=BUCKET_SIZE,
                                     batch_size=batch_size, droplast=is_train)

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num)
    return loader


def build_dataloader_itm_ms(dataset, collate_fn, is_train, opts, device_num):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketPathSamplerForItm(dataset, bucket_size=BUCKET_SIZE,
                                           batch_size=batch_size, droplast=is_train)

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num)
    return loader


def build_dataloader_ret(dataset, collate_fn, device_num, batch_size=4):
    sampler = BatchSampler(len(dataset), batch_size=batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num)
    return loader


def build_dataloader_cap(dataset, collate_fn, device_num, batch_size=4):
    sampler = BatchSampler(len(dataset), batch_size=batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num)
    return loader


# Masked Language Modeling
def build_mlmThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MlmThreeDataset(ids_path, txt_db, img_db, audio_db, use_video=opts.use_video,
                              use_mask_fix=opts.use_mask_fix)
    return dataset, mlmThree_collate


# Masked Region Classification
def build_mrcThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MrcThreeDataset(ids_path, opts.mrm_prob, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, mrcThree_collate


# Masked Region Feature Regression (MRFR)
def build_mrfrThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MrfrThreeDataset(opts.mrm_prob, ids_path, txt_db, img_db, audio_db, use_video=opts.use_video,
                               use_mask_fix=opts.use_mask_fix)
    return dataset, mrfrThree_collate


# Masked Audio Feature Regression (MAFR)
def build_mafrThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MafrThreeDataset(opts.mrm_prob, ids_path, txt_db, img_db, audio_db, use_video=opts.use_video,
                               use_mask_fix=opts.use_mask_fix)
    return dataset, mafrThree_collate


# (ITM)
def build_itmThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = ItmThreeDataset(ids_path, txt_db, img_db, audio_db, opts.itm_neg_prob, use_video=opts.use_video,
                              use_mask_fix=opts.use_mask_fix)
    return dataset, itmThree_collate


# Text Output
def build_tdThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TdThreeDataset(ids_path, txt_db, img_db, audio_db, use_video=opts.use_video)
    return dataset, tdThree_collate


# Image Output
def build_idThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = IdThreeDataset(ids_path, txt_db, img_db, audio_db, opts.img_token_path, data_type=opts.data_type)
    return dataset, idThree_collate


# retrieval dataset
def build_ti2a_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TxtImgtoAudioEvalDataset(ids_path, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, itmMatchingThree_collate


def build_t2i_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TxttoImgEvalDataset(ids_path, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, itmMatchingTxtImg_collate


def build_t2a_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TxttoAudioEvalDataset(ids_path, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, itmMatchingTxtAudio_collate


def build_ta2i_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TxtAudiotoImgEvalDataset(ids_path, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, itmMatchingThree_collate


def build_a2i_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = AudiotoImgEvalDataset(ids_path, txt_db, img_db, audio_db)
    return dataset, itmMatchingThree_collate


# Text One Decode Output DataLoader
def create_tdOne_dataloader(ids_path, img_dir, opts, device_num):
    dataset = TdOneDataset(ids_path, img_dir)
    loader = build_dataloader_ms(dataset, tdOne_collate, False, opts, device_num)
    dataloaders = {}
    dataloaders['tdOne'] = loader
    return dataloaders


# def build_adText_dataset_v3(ids_path, txt_db, img_db, audio_db, opts):
#     preprocess_config = yaml.load(open(opts.audio_preprocess_config, "r"), Loader=yaml.FullLoader)
#     dataset = AdTextV3Dataset(ids_path, txt_db, opts.audio_mel_path, preprocess_config)
#     return dataset, adTextV3_collate


def create_three_dataloaders(ids_path, datasets, is_train, opts, device_num, ids_two_path=None,
                             ids_textaudio_path=None):
    """ Create dataloaders """
    dataloaders = {}
    ## finetune Retrieval
    dset = datasets[0]
    if dset['tasks'][0].startswith('ftRet'):
        txt_db = TxtTokThreeLmdb(dset['db'][0], use_video=opts.use_video, data_type=opts.data_type, name=opts.name_txt)
        img_db = DetectFeatThreeLmdb(dset['img'][0], dset['db'][0], use_video=opts.use_video, data_type=opts.data_type,
                                     name=opts.name_img)
        dataset = ItmFlickrRankDataset(ids_path, txt_db, img_db, neg_sample_size=1)
        loader = build_dataloader_ret(dataset, itm_rank_collate, device_num, batch_size=10)
        dataloaders["ftRet"] = loader
        return dataloaders

    if dset['tasks'][0].startswith('ftCap'):
        txt_db = TxtTokThreeLmdb(dset['db'][0], use_video=opts.use_video, data_type=opts.data_type, name=opts.name_txt)
        img_db = DetectFeatThreeLmdb(dset['img'][0], dset['db'][0], use_video=opts.use_video, data_type=opts.data_type,
                                     name=opts.name_img)
        dataset = CaptionDataset(ids_path, txt_db, img_db)

        batch_size = opts.train_batch_size if is_train else opts.val_batch_size

        loader = build_dataloader_cap(dataset, caption_collate, device_num, batch_size=batch_size)

        dataloaders["ftCap"] = loader
        return dataloaders

    for dset in datasets:
        if dset['tasks']:  # if the list sequence is empty, then it is equal to False
            txt_db = TxtTokThreeLmdb(dset['db'][0], use_video=opts.use_video, data_type=opts.data_type,
                                     name=opts.name_txt)
            img_db = DetectFeatThreeLmdb(dset['img'][0], dset['db'][0], use_video=opts.use_video,
                                         data_type=opts.data_type, name=opts.name_img)
            audio_db = AudioFeatThreeLmdb(dset['audio'][0], dset['db'][0], use_video=opts.use_video,
                                          data_type=opts.data_type, name=opts.name_audio)

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'
            if task.startswith('mlmThree'):
                dataset, collate_fn = build_mlmThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mrcThree'):
                dataset, collate_fn = build_mrcThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mrfrThree'):
                dataset, collate_fn = build_mrfrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mrctThree'):
                dataset, collate_fn = build_mrfrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('itmThree'):
                dataset, collate_fn = build_itmThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mafrThree'):
                dataset, collate_fn = build_mafrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('macThree'):
                dataset, collate_fn = build_mafrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('tdThree'):
                dataset, collate_fn = build_tdThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('idThree'):
                dataset, collate_fn = build_idThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('ret_t2i'):
                dataset, collate_fn = build_t2i_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('ret_ti2a'):
                dataset, collate_fn = build_ti2a_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('ret_t2a'):
                dataset, collate_fn = build_t2a_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('ret_ta2i'):
                dataset, collate_fn = build_ta2i_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('ret_a2i'):
                dataset, collate_fn = build_a2i_dataset(ids_path, txt_db, img_db, audio_db, opts)
            else:
                raise ValueError('Undefined task %s' % (task))
            LOGGER.info("Create Dataset %s Success", (task))
            if task.startswith('itm'):
                # itm handles distributed training in dset not sampler
                loader = build_dataloader_itm_ms(dataset, collate_fn, is_train, opts, device_num)
            elif task.startswith('ret') or task.startswith('td'):
                loader = build_dataloader_ret(dataset, collate_fn, device_num)
            else:
                loader = build_dataloader_ms(dataset, collate_fn, is_train, opts, device_num)

            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = loader
    return dataloaders
