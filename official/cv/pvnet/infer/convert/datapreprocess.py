# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import shutil
import numpy as np
from model_utils.config import config as cfg
from model_utils.data_file_utils import read_pickle


if __name__ == '__main__':
    re_dir = os.path.join("./data/", cfg.cls_name)
    img_dir = os.path.join(re_dir, 'images')
    pose_dir = os.path.join(re_dir, 'poses')
    # 6D pose estimation input
    real_pkl = os.path.join(cfg.eval_dataset, cfg.dataset_name, 'posedb', '{}_real.pkl'.format(cfg.cls_name))
    real_set = read_pickle(real_pkl)

    data_root_dir = os.path.join(cfg.dataset_dir, cfg.dataset_name, cfg.cls_name)
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

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)

    for idx, _ in enumerate(test_db):

        rgb_path = os.path.join(cfg.eval_dataset, cfg.dataset_name, test_db[idx]['rgb_pth'])
        rgb_name = test_db[idx]['rgb_pth'].strip().split('/')[-1]
        shutil.copyfile(rgb_path, os.path.join(img_dir, rgb_name))

        pose = test_db[idx]['RT'].copy()
        np.savetxt(os.path.join(pose_dir, rgb_name.split('.')[0]+'.txt'), pose)
    np.savetxt(os.path.join(re_dir, "test.txt"), test_fns, fmt='%s')
    print('preprocess success!')
