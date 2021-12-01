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
""" data_three """

import os
import io
import json
import gc
import lmdb
from config import config
from tqdm import tqdm
import numpy as np
from mindspore.communication.management import get_group_size, get_rank
from data.data import TxtLmdb
from .utils import pad_sequence

global_ids = {}


class DetectFeatThreeLmdb():
    """ DetectFeatThreeLmdb """

    def __init__(self, img_dir, txt_dir, name="img2len.json", use_lmdb=True, data_ver=0, use_video=False, data_type=0):
        self.img_dir = img_dir
        self.data_ver = data_ver
        self.use_video = use_video
        self.data_type = data_type

        if use_lmdb:
            self.img_dir_npz = self.img_dir + "_npz"
        else:
            self.img_dir_npz = self.img_dir

        self.use_lmdb = use_lmdb
        if os.path.exists(self.img_dir_npz):
            self.use_lmdb = False

        print("{} use_lmdb {}".format(self.img_dir_npz, self.use_lmdb))
        if self.use_lmdb:
            self.env = lmdb.open(f'{self.img_dir}',
                                 readonly=True, create=False)
            self.txn = self.env.begin(buffers=True)

        path_img2len = os.path.join(txt_dir, name)
        with open(path_img2len) as f:
            tmp_name2len = json.load(f)

        self.name2len = {}
        for key, val in tmp_name2len.items():
            self.name2len[key] = min(val, config.MAX_IMG_LEN)

    def get_video(self, id_):
        """ get_video """

        if self.use_lmdb:
            data = self.txn.get(id_.encode('utf-8'))
            data = np.load(io.BytesIO(data))
        else:
            if id_.endswith(".npz"):
                name = id_
            else:
                name = id_ + ".npz"
            data = np.load(os.path.join(self.img_dir_npz, name[:2], name))

        np_att_feat = data['feat']

        att_feat = np.array(np_att_feat).astype(np.float32)
        att_feat = att_feat[:config.MAX_IMG_LEN, :]

        return att_feat

    def __getitem__(self, id_):

        if self.use_video:
            return self.get_video(id_)

        if self.data_type == 1:
            feat_path = os.path.join(self.img_dir_npz, id_.replace("_", "/").replace(".jpg", ".npz"))
            try:
                data = np.load(feat_path)
            except FileNotFoundError:
                print("err file: " + feat_path)
        elif self.data_type == 2:
            if ".jpg" in id_:
                num = id_.split(".jpg")[-1]
                feat_path = os.path.join(self.img_dir_npz, id_.replace(".jpg" + num, ".jpg.npz"))
                if not os.path.exists(feat_path):
                    feat_path = os.path.join(self.img_dir_npz, id_.replace(".jpg" + num, ".npz"))
            else:
                feat_path = os.path.join(self.img_dir_npz, id_ + ".npz")
            try:
                data = np.load(feat_path)
            except FileNotFoundError:
                print("err file: " + feat_path)
        elif self.use_lmdb:
            data = self.txn.get(id_.encode('utf-8'))
            data = np.load(io.BytesIO(data))
        else:
            if id_.endswith(".npz"):
                name = id_
            else:
                name = id_ + ".npz"
            data = np.load(os.path.join(self.img_dir_npz, name))
            if self.data_ver == 1:
                data1 = np.load(os.path.join(self.img_dir_npz.replace("_att", "_info"), name))

        np_att_feat = data['feat']
        if self.data_ver == 1:
            data = data1
        np_pred_boxes = data['pred_boxes']
        np_scores = data['scores']
        np_pred_classes = data['pred_classes']
        np_width = data['width']
        np_height = data['height']

        att_feat = np.array(np_att_feat).astype(np.float32)
        att_feat = att_feat[:config.MAX_IMG_LEN, :]

        # x1, y1, x2, y2
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

        return att_feat, pred_boxes, scores, pred_classes

    def __del__(self):
        if self.use_lmdb:
            self.env.close()


class AudioFeatThreeLmdb():
    """ AudioFeatThreeLmdb """

    def __init__(self, audio_dir, txt_dir, use_video=False, data_type=0, name="audio2len.json"):
        self.audio_dir = audio_dir
        self.txt_dir = txt_dir
        self.use_video = use_video
        self.data_type = data_type

        self.audio_dir_npz = self.audio_dir + "_npz"
        print(self.audio_dir_npz)
        # assert os.path.exists(self.audio_dir) or os.path.exists(self.audio_dir_npz)

        # only read ahead on single node training
        self.use_lmdb = not use_video
        if os.path.exists(self.audio_dir_npz):
            self.use_lmdb = False

        print("{} use_lmdb {}".format(self.audio_dir_npz, self.use_lmdb))

        if self.use_lmdb:
            self.env = lmdb.open(f'{self.audio_dir}',
                                 readonly=True, create=False)
            self.txn = self.env.begin(buffers=True)

        path_audio2len = os.path.join(txt_dir, name)
        with open(path_audio2len) as f:
            tmp_name2len = json.load(f)

        self.name2len = {}
        for key, val in tmp_name2len.items():
            self.name2len[key] = min(val, config.MAX_AUDIO_LEN)

    def __del__(self):
        if self.use_lmdb:
            self.env.close()

    def __getitem__(self, file_name):
        if self.data_type == 1:
            path_npz = os.path.join(self.audio_dir_npz, file_name.replace("_", "/") + ".npz")
            feat = np.load(path_npz)['feat']
        elif self.data_type == 2:
            if "cc3m" in file_name:
                file_name = file_name.replace("/training", "").replace("/validation", "")
            path_npz = os.path.join(self.audio_dir_npz, file_name + ".npz")
            feat = np.load(path_npz)['feat']
        elif self.use_video:
            path_npy = os.path.join(self.audio_dir, file_name + ".npy")
            feature = np.load(path_npy)
            feat = np.array(feature.T.reshape(-1, 512))
        elif self.use_lmdb:
            data = self.txn.get(file_name.encode('utf-8'))
            feature = np.frombuffer(data, dtype=np.float32)
            feat = np.array(feature.T.reshape(-1, 512))
        else:
            path_npz = os.path.join(self.audio_dir_npz, file_name + ".npz")
            feat = np.load(path_npz)['feat']
        # T * 512
        audio_feat = np.array(feat).astype(np.float32)
        audio_feat = audio_feat[:config.MAX_AUDIO_LEN, :]
        return audio_feat


# Text Data
# 300: 0.904767
class TxtTokThreeLmdb():
    """ TxtTokThreeLmdb """

    def __init__(self, db_dir, use_audio=True, use_video=False, data_type=0, name="id2len.json"):

        print("TxtTokThreeLmdb {}".format(db_dir))
        with open(f'{db_dir}/{name}') as f:
            id2len = json.load(f)

        self.use_video = use_video
        self.data_type = data_type

        self.id2len = {}
        for id_, len_ in id2len.items():
            self.id2len[id_] = len_

        self.db_dir = db_dir
        self.db_dir_json = self.db_dir + "_json"
        self.use_lmdb = True

        if os.path.exists(self.db_dir_json):
            self.use_lmdb = False

        print("{} use_lmdb {}".format(self.db_dir_json, self.use_lmdb))

        if self.use_lmdb:
            self.db = TxtLmdb(db_dir, readonly=True)
        with open(f'{db_dir}/meta.json', 'r') as f:
            meta = json.load(f)
        # meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        if self.data_type == 1:
            path_json = os.path.join(self.db_dir_json, id_.replace("_", "/") + ".json")
            with open(path_json) as f:
                txt_dump = json.load(f)
        elif self.data_type == 2:
            path_json = os.path.join(self.db_dir_json, id_ + ".json")
            with open(path_json) as f:
                txt_dump = json.load(f)
        elif self.use_video:
            path_json = os.path.join(self.db_dir_json, id_[:2], id_ + ".json")
            with open(path_json) as f:
                txt_dump = json.load(f)
        elif self.use_lmdb:
            txt_dump = self.db[id_]
        else:
            path_json = os.path.join(self.db_dir_json, id_ + ".json")
            with open(path_json) as f:
                txt_dump = json.load(f)

        return txt_dump

    @property
    def txt2img(self):
        with open(f'{self.db_dir}/txt2img.json') as f:
            txt2img = json.load(f)
        return txt2img

    @property
    def txt2audio(self):
        with open(f'{self.db_dir}/txt2audio.json') as f:
            txt2audio = json.load(f)
        return txt2audio

    @property
    def img2txts(self):
        with open(f'{self.db_dir}/img2txts.json') as f:
            img2txts = json.load(f)
        return img2txts

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return np.array(input_ids)


# get distribute data
def get_ids_and_lens_three(db):
    assert isinstance(db, TxtTokThreeLmdb)
    lens = []
    ids = []
    for id_ in list(db.id2len.keys())[get_rank()::get_group_size()]:
        lens.append(db.id2len[id_])
        ids.append(id_)
    return lens, ids


def get_ids_three(ids_path):
    ids = json.load(open(ids_path))
    size, rank = get_size_rank()
    return ids[rank::size]


def get_size_rank():
    size, rank = 1, 0
    return size, rank


def get_tmp_path_(ids_path):
    size, rank = 1, 0
    tmp_path = "{}_{}_{}.json".format(ids_path, size, rank)
    return tmp_path


def get_tmp_ids_path(ids_path):
    if not os.path.exists(os.path.dirname(ids_path) + "/temp"):
        os.mkdir(os.path.dirname(ids_path) + "/temp")
    tmp_ids_path = os.path.dirname(ids_path) + "/temp/" + os.path.basename(ids_path)
    return tmp_ids_path


def get_tmp_path_with_name(ids_path, name):
    size, rank = get_size_rank()
    tmp_ids_path = get_tmp_ids_path(ids_path)
    tmp_path = "{}_{}_{}_{}.json".format(tmp_ids_path, size, rank, name)
    return tmp_path


def get_tmp_two_path_(ids_path):
    size, rank = get_size_rank()
    tmp_path = "{}_{}_{}_two.json".format(ids_path, size, rank)
    return tmp_path


def get_ids_three_(ids_path):
    size, rank = get_size_rank()
    ids = json.load(open(ids_path))
    tmp_path = get_tmp_path_(ids_path)
    return ids[rank::size], tmp_path


def get_data_id_(ids_path):
    size, rank = get_size_rank()
    id_ = "{}_{}_{}".format(ids_path, size, rank)
    return id_


def pad_sequence_(sequences, batch_first=False, padding_value=0.0):
    if sequences[0] is None:
        return None
    return pad_sequence(sequences, batch_first, padding_value)


# Image feature, Text token, Audio feature
class DetectFeatTxtTokAudioFeatDataset():
    """ DetectFeatTxtTokAudioFeatDataset """

    def __init__(self, ids_path, txt_db, img_db, audio_db, use_video=False):

        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)
        assert isinstance(audio_db, AudioFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_video = use_video

        global global_ids

        id_ = get_data_id_(ids_path)
        if id_ in global_ids:
            self.ids = global_ids[id_]
            print("data use global_ids {} ids:{}".format(id_, len(self.ids)))
        else:
            self.ids = get_ids_three(ids_path)
            global_ids[id_] = self.ids
            print("data generate global_ids {} ids:{}".format(id_, len(self.ids)))

        self.path_lens = get_tmp_path_(ids_path)

        if not os.path.exists(self.path_lens):
            print("data generate")
            self.lens = []
            for id_ in tqdm(self.ids, 'compute len'):
                txt_len = min(config.MAX_TEXT_LEN, self.txt_db.id2len[id_])
                img_len = min(config.MAX_IMG_LEN, self.img_db.name2len[id_])
                audio_len = min(config.MAX_AUDIO_LEN, self.audio_db.name2len[id_])
                bert_len = txt_len + img_len + audio_len
                self.lens.append(bert_len)
            json.dump(self.lens, open(self.path_lens, "w"))
            del self.lens
            gc.collect()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        example['id'] = id_
        example['img_fname'] = id_
        example['audio_fname'] = id_
        return example

    def _get_img_feat(self, id_):
        if self.use_video:
            img_feat = self.img_db.get_video(id_)
            bb_num = img_feat.shape[0]
            img_bb = None
        else:
            img_feat, pred_boxes, _, _ = self.img_db[id_]
            img_bb = pred_boxes
            bb_num = img_feat.shape[0]
        return img_feat, img_bb, bb_num

    def _get_audio_feat(self, id_):
        audio_feat = self.audio_db[id_]
        return audio_feat, audio_feat.shape[0]

    def _get_txt_token(self, input_ids):
        input_ids = input_ids[: config.MAX_TEXT_GTS_LEN]
        txt_inputs = np.array([0] + input_ids)
        txt_gts = np.array(input_ids + [0])
        txt_masks = np.ones(len(txt_gts))
        return txt_inputs, txt_gts, txt_masks


# Image feature, Text token, Audio feature
class DetectFeatTxtTokTwoDataset():
    """ DetectFeatTxtTokTwoDataset """

    def __init__(self, ids_path, txt_db, img_db, use_video=False):

        assert isinstance(txt_db, TxtTokThreeLmdb)
        assert isinstance(img_db, DetectFeatThreeLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.use_video = use_video

        global global_ids

        id_ = get_data_id_(ids_path)
        if id_ in global_ids:
            self.ids = global_ids[id_]
            print("data use global_ids {} ids:{}".format(id_, len(self.ids)))
        else:
            self.ids = get_ids_three(ids_path)
            global_ids[id_] = self.ids
            print("data generate global_ids {} ids:{}".format(id_, len(self.ids)))

        self.path_lens = get_tmp_path_(ids_path)

        if not os.path.exists(self.path_lens):
            print("data generate")
            self.lens = []
            for id_ in tqdm(self.ids, 'compute len'):
                txt_len = min(config.MAX_TEXT_LEN, self.txt_db.id2len[id_])
                img_len = min(config.MAX_IMG_LEN, self.img_db.name2len[id_])
                bert_len = txt_len + img_len
                self.lens.append(bert_len)
            json.dump(self.lens, open(self.path_lens, "w"))
            del self.lens
            gc.collect()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        example['id'] = id_
        example['img_fname'] = id_
        return example

    def _get_img_feat(self, id_):
        if self.use_video:
            img_feat = self.img_db.get_video(id_)
            bb_num = img_feat.shape[0]
            img_bb = None
        else:
            img_feat, pred_boxes, _, _ = self.img_db[id_]
            img_bb = pred_boxes
            bb_num = img_feat.shape[0]
        return img_feat, img_bb, bb_num

    def _get_txt_token(self, input_ids):
        input_ids = input_ids[: config.MAX_TEXT_GTS_LEN]
        txt_inputs = np.array([0] + input_ids)
        txt_gts = np.array(input_ids + [0])
        txt_masks = np.ones(len(txt_gts))
        return txt_inputs, txt_gts, txt_masks


def get_gather_index_three(txt_lens, num_bbs, num_aus, batch_size, max_len, max_bb, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    for i, (tl, nbb, nau) in enumerate(zip(txt_lens, num_bbs, num_aus)):
        gather_index[i, tl:tl + nbb] = np.arange(max_len, max_len + nbb, dtype=np.int64)
        gather_index[i, tl + nbb:tl + nbb + nau] = np.arange(max_len + max_bb, max_len + max_bb + nau, dtype=np.int64)

    return gather_index


global_ids_text = {}


class TxtTokDataset():
    """ TxtTokDataset """

    def __init__(self, ids_path, txt_db):
        assert isinstance(txt_db, TxtTokThreeLmdb)
        self.txt_db = txt_db
        self.ids_path = ids_path

        self.compute_lens()

    def compute_lens(self):
        """ compute_lens """

        ids_path = self.ids_path
        global global_ids_text

        id_ = get_data_id_(ids_path)
        if id_ in global_ids_text:
            self.ids = global_ids_text[id_]
            print("data use global_ids {} ids:{}".format(id_, len(self.ids)))
        else:
            self.ids = get_ids_three(ids_path)
            global_ids_text[id_] = self.ids
            print("data generate global_ids {} ids:{}".format(id_, len(self.ids)))

        self.path_lens = get_tmp_path_with_name(ids_path, "text")

        if not os.path.exists(self.path_lens):
            print("data generate")
            self.lens = []
            for id_ in tqdm(self.ids, 'compute len'):
                txt_len = min(config.MAX_TEXT_LEN, self.txt_db.id2len[id_])
                bert_len = txt_len
                self.lens.append(bert_len)
            json.dump(self.lens, open(self.path_lens, "w"))
            del self.lens
            gc.collect()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        i = i % len(self.ids)
        id_ = self.ids[i]
        example = self.txt_db[id_]
        example['id'] = id_
        example['img_fname'] = id_
        example['audio_fname'] = id_
        return example
