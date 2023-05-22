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
"""Convert Image to MindRecord"""
import os

import numpy as np
from mindspore.mindrecord import FileWriter

from data_file_utils import read_rgb_np, read_mask_np, read_pickle


linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone',
                     'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']


def data2mindrecord(cls_name, mindrecord_dir, pvnet_path):
    """train data to mind record format"""
    if not os.path.exists(mindrecord_dir):
        os.mkdir(mindrecord_dir)
    MINDRECORD_FILE = os.path.join(mindrecord_dir, "pvnettrain.mindrecord")
    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)

    writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=8)
    vote_num = 9
    cv_schema = {"image": {"type": "bytes"},
                 "mask": {"type": "bytes"},
                 "farthest": {"type": "float64", "shape": [-1, vote_num * 2]}
                 }
    writer.add_schema(cv_schema, "cv_schema")
    save_data = []
    linemod_dir = os.path.join(pvnet_path, 'LINEMOD')
    # render dataset
    render_pkl = os.path.join(linemod_dir, 'posedb', '{}_render.pkl'.format(cls_name))
    render_set = read_pickle(render_pkl)
    # real train dataset
    train_real_set = []
    test_real_set = []
    val_real_set = []
    real_pkl = os.path.join(linemod_dir, 'posedb', '{}_real.pkl'.format(cls_name))
    real_set = read_pickle(real_pkl)
    train_fn = os.path.join(linemod_dir, '{}/train.txt'.format(cls_name))
    test_fn = os.path.join(linemod_dir, '{}/test.txt'.format(cls_name))
    val_fn = os.path.join(linemod_dir, '{}/val.txt'.format(cls_name))

    with open(test_fn, 'r') as f:
        test_fns = [line.strip().split('/')[-1] for line in f.readlines()]

    with open(train_fn, 'r') as f:
        train_fns = [line.strip().split('/')[-1] for line in f.readlines()]

    with open(val_fn, 'r') as f:
        val_fns = [line.strip().split('/')[-1] for line in f.readlines()]

    for data in real_set:
        if data['rgb_pth'].split('/')[-1] in test_fns:
            if data['rgb_pth'].split('/')[-1] in val_fns:
                val_real_set.append(data)
            else:
                test_real_set.append(data)

        if data['rgb_pth'].split('/')[-1] in train_fns:
            train_real_set.append(data)

    # fuse dataset
    fuse_pkl = os.path.join(linemod_dir, 'posedb', '{}_fuse.pkl'.format(cls_name))
    fuse_set = read_pickle(fuse_pkl)

    train_db = []
    train_db += render_set
    train_db += train_real_set
    train_db += fuse_set

    dataset_size = len(train_db)
    for i in range(dataset_size):
        i += 1
        print("Processing {}/{}".format(i, dataset_size))

        img = os.path.join(linemod_dir, train_db[i - 1]['rgb_pth'])
        mask = os.path.join(linemod_dir, train_db[i - 1]['dpt_pth'])

        rgb = read_rgb_np(img)
        mask = read_mask_np(mask)

        if train_db[i - 1]['rnd_typ'] == 'real' and len(mask.shape) == 3:
            mask = np.sum(mask, 2) > 0
            mask = np.asarray(mask, np.int32)

        if train_db[i - 1]['rnd_typ'] == 'fuse':
            mask = np.asarray(mask == (linemod_cls_names.index(train_db[i - 1]['cls_typ']) + 1), np.int32)

        cen = train_db[i - 1]['center'].copy()
        far = train_db[i - 1]['farthest'].copy()
        hcoords = np.concatenate([far, cen], 0)

        sample = {}

        sample['image'] = rgb.tobytes()
        sample['mask'] = mask.tobytes()
        sample['farthest'] = hcoords

        save_data.append(sample)

        if i % 1000 == 0:
            writer.write_raw_data(save_data)
            save_data = []

    if save_data:
        writer.write_raw_data(save_data)
    writer.commit()


if __name__ == "__main__":
    cls_list = ["cat"]
    for item in cls_list:
        data2mindrecord(item, "/cache/{}".format(item), "/data/bucket-4609/dataset/pvnet/data")
