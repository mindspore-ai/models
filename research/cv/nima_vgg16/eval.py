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
"""Evaluate model."""

import numpy as np
import scipy.stats
from mindspore import context
from mindspore import load_checkpoint

from src.MyDataset import create_dataset
from src.config import config
from src.vgg import vgg16


if __name__ == "__main__":
    args = config
    if args.enable_modelarts:
        import moxing as mox
        mox.file.shift('os', 'mox')
    config.device_num = 1
    config.rank = config.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False,
                        device_id=args.device_id, reserve_class_name_in_scope=False)
    ds_val, steps_per_epoch_val = create_dataset(args, data_mode='val')
    net = vgg16(10, args=config, phase='test')
    net.set_train(False)
    load_checkpoint(args.ckpt_file, net=net)
    total_score = []
    total_gt = []
    SCORE_LIST = np.array([x for x in range(1, 11)])
    for i, (data, gt_classes) in enumerate(ds_val):
        gt_classes = gt_classes.asnumpy()
        output = net(data)
        output = output.asnumpy()
        gt = np.sum(gt_classes * np.array(SCORE_LIST), axis=1)
        score = np.sum(output * np.array(SCORE_LIST), axis=1)
        total_score += score.tolist()
        total_gt += gt.tolist()
    total_score = np.array(total_score)
    total_gt = np.array(total_gt)
    print('mse:', np.mean(np.power((total_score-total_gt), 2)))
    print('deal imgs is:', total_score.shape[0])
    print('SRCC:', scipy.stats.spearmanr(total_score, total_gt)[0])
