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
import numpy as np
import libransac_voting as ransac_vote
from model_utils.config import config as cfg
from src.evaluation_utils import Evaluator


def acc(test_fn, pose_dir, result_dir):
    channel = cfg.vote_num * 2 + 2
    ransac_vote.init_voting(cfg.img_height, cfg.img_width, channel, 2, cfg.vote_num)
    test_fns = np.loadtxt(test_fn, dtype=str)
    evaluator = Evaluator()
    for _, img_fn in enumerate(test_fns):
        seg_fn = os.path.join(result_dir, 'seg_pred', img_fn.split('.')[0] + '.bin')
        ver_fn = os.path.join(result_dir, 'ver_pred', img_fn.split('.')[0] + '.bin')
        seg_pred = np.fromfile(seg_fn, dtype=np.float32).reshape(1, -1, 480, 640)
        ver_pred = np.fromfile(ver_fn, dtype=np.float32).reshape(1, -1, 480, 640)
        pose = np.loadtxt(os.path.join(pose_dir, img_fn.split('.')[0] + '.txt')).reshape(3, 4)

        data = np.concatenate([seg_pred, ver_pred], 1)[0]
        corner_pred = np.zeros((cfg.vote_num, 2), dtype=np.float32)
        ransac_vote.do_voting(data, corner_pred)
        pose_pred = evaluator.evaluate(corner_pred, pose, cfg.cls_name)
        np.savetxt(os.path.join(result_dir, "pred_pose", img_fn.split('.')[0] + '.txt'), pose_pred)

    proj_err, add, _ = evaluator.average_precision(False)
    print('Processing object:{}, 2D error:{}, ADD:{}'.format(cfg.cls_name, proj_err, add))


if __name__ == '__main__':
    test_fn_out = os.path.join('./data', cfg.cls_name, 'test.txt')
    pose_dir_out = os.path.join('./data', cfg.cls_name, 'poses')
    result_dir_out = './result'
    if not os.path.exists(os.path.join(result_dir_out, 'pred_pose')):
        os.makedirs(os.path.join(result_dir_out, 'pred_pose'))

    acc(test_fn=test_fn_out, pose_dir=pose_dir_out, result_dir=result_dir_out)
