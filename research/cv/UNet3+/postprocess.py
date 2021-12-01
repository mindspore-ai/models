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
'''postprocess'''
import os
import numpy as np

from src.config import config as cfg
from eval import dice_coef, AverageMeter

def sigmoid(z):
    '''define sigmoid'''
    return 1/(1 + np.exp(-z))

def cal_acc():
    '''calculate accuracy'''
    all_mask = np.load(cfg.label_path)
    file_num = len(os.listdir(cfg.post_result_path))
    bs = cfg.export_batch_size
    dices = AverageMeter()

    for i in range(file_num):
        file_name = "unet3p_bs" + str(bs) + "_" + str(i) + "_0.bin"
        mask = all_mask[i]
        result_f = np.fromfile(os.path.join(cfg.post_result_path, file_name), np.float32)
        result_f = result_f.reshape(bs, 1, cfg.image_height, cfg.image_width)
        output = sigmoid(result_f)
        dice = dice_coef(output, mask)
        dices.update(dice, bs)
    print("Final dices: ", str(dices.avg))

if __name__ == '__main__':
    cal_acc()
