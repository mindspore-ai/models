# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""eval"""
import os
import time

import numpy as np
import mindspore
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as V
from mindspore import context

from model_utils.config import config as cfg
from model_utils.data_file_utils import read_pickle, read_rgb_np
from src.evaluation_utils import Evaluator
from src.lib.voting import libransac_voting as ransac_vote
from src.model_reposity import Resnet18_8s


def test(args):
    """eval"""
    cls_list = args.cls_name
    seg_dim = 1 + len(cls_list.split(','))

    # only support single obj for current
    assert seg_dim == 2

    # set graph mode and parallel mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.rank)

    # load model parameters
    net = Resnet18_8s(ver_dim=args.vote_num * 2)
    param_dict = mindspore.load_checkpoint(args.ckpt_file)
    mindspore.load_param_into_net(net, param_dict)
    net.set_train(False)
    print("---------  test is starting  ---------")
    real_pkl = os.path.join(args.eval_dataset, args.dataset_name, 'posedb', '{}_real.pkl'.format(args.cls_name))
    real_set = read_pickle(real_pkl)

    data_root_dir = os.path.join(args.dataset_dir, args.dataset_name, args.cls_name)
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

    evaluator = Evaluator()
    total_infer_time = 0.0
    channel = args.vote_num * 2 + 2
    ransac_vote.init_voting(args.img_height, args.img_width, channel, 2, args.vote_num)
    # for item in data_list:
    for idx, _ in enumerate(test_db):
        rgb_path = os.path.join(args.eval_dataset, args.dataset_name, test_db[idx]['rgb_pth'])
        pose = test_db[idx]['RT'].copy()

        rgb = read_rgb_np(rgb_path)
        rgb = V.ToTensor()(rgb)
        rgb = C.TypeCast(mindspore.dtype.float32)(rgb)
        # Computed from random subset of ImageNet training images
        rgb = V.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)(rgb)
        rgb = np.expand_dims(rgb, axis=0)
        rgb = mindspore.Tensor(rgb)

        start = time.time()
        seg_pred, ver_pred = net(rgb)
        infer_cost = (time.time() - start) * 1000
        total_infer_time += infer_cost

        seg_pred = seg_pred.asnumpy()
        ver_pred = ver_pred.asnumpy()
        data = np.concatenate([seg_pred, ver_pred], 1)[0]

        corner_pred = np.zeros((args.vote_num, 2), dtype=np.float32)
        ransac_vote.do_voting(data, corner_pred)
        evaluator.evaluate(corner_pred, pose, args.cls_name)

        print('Processing object:{}, image numbers:{}/{}, inference time:{}ms'.format(cls_list,
                                                                                      idx + 1, len(test_db),
                                                                                      infer_cost))
    proj_err, add, _ = evaluator.average_precision(False)
    print('Processing object:{}, 2D error:{}, ADD:{}, inference cost:{}ms'.format(cls_list, proj_err, add,
                                                                                  total_infer_time / len(test_db)))

    print("--------- test is finished ---------")


if __name__ == '__main__':
    test(cfg)
