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
"""evaluate postprocess"""
import os

import numpy as np

from model_utils.config import config as cfg
from model_utils.data_file_utils import read_pickle
from src.evaluation_utils import Evaluator
from src.lib.voting import libransac_voting as ransac_vote


def test(args):
    """postprocess: do ransac voting"""
    print("--------- test is starting ---------")
    print(args)
    real_pkl = os.path.join(args.eval_dataset, args.dataset_name, 'posedb', '{}_real.pkl'.format(args.cls_name))
    real_set = read_pickle(real_pkl)

    data_root_dir = os.path.join(args.eval_dataset, args.dataset_name, args.cls_name)
    test_fn = os.path.join(data_root_dir, 'test.txt')
    val_fn = os.path.join(data_root_dir, 'val.txt')

    with open(test_fn, 'r') as f:
        test_fns = [line.strip().split('/')[-1] for line in f.readlines()]
    with open(val_fn, 'r') as f:
        val_fns = [line.strip().split('/')[-1] for line in f.readlines()]

    test_real_set = []
    val_real_set = []

    for data in real_set:
        if data['rgb_pth'].split('/')[-1] in test_fns:
            if data['rgb_pth'].split('/')[-1] in val_fns:
                val_real_set.append(data)
            else:
                test_real_set.append(data)

    test_db = []
    test_db += test_real_set
    test_db += val_real_set

    seg_pred_shape = (2, args.img_height, args.img_width)
    ver_pred_shape = (args.vote_num * 2, args.img_height, args.img_width)

    evaluator = Evaluator()

    channel = args.vote_num * 2 + 2
    ransac_vote.init_voting(args.img_height, args.img_width, channel, 2, args.vote_num)
    # for item in data_list:
    for idx, _ in enumerate(test_db):
        rgb_path = os.path.join(args.eval_dataset, args.dataset_name, test_db[idx]['rgb_pth'])
        pose = test_db[idx]['RT'].copy()
        rgb_idx = rgb_path.split('.jpg')[0].split('/')[-1]

        seg_pred_path = os.path.join(args.result_path, rgb_idx + '_0.bin')
        seg_pred = np.fromfile(seg_pred_path, dtype=np.float32).reshape(seg_pred_shape)
        ver_pred_path = os.path.join(args.result_path, rgb_idx + '_1.bin')
        ver_pred = np.fromfile(ver_pred_path, dtype=np.float32).reshape(ver_pred_shape)

        data = np.concatenate([seg_pred, ver_pred], 0)
        corner_pred = np.zeros((args.vote_num, 2), dtype=np.float32)
        ransac_vote.do_voting(data, corner_pred)
        evaluator.evaluate(corner_pred, pose, args.cls_name)

        print('Processing object:{}, image numbers:{}/{}'.format(args.cls_name, idx + 1, len(test_db)))
    proj_err, add, _ = evaluator.average_precision(False)
    print('Processing object:{}, 2D projection error:{}, ADD:{}'.format(args.cls_name, proj_err, add))
    print("--------- test is finished ---------")


if __name__ == '__main__':
    test(cfg)
