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
"""get dataset loader"""
import os
import math
import time
import SimpleITK as sitk
import numpy as np


def file_name(file_dir):
    L = []
    path_list = os.listdir(file_dir)
    path_list.sort()
    for filename in path_list:
        if 'test' in filename and 'mhd' in filename:
            L.append(os.path.join(filename))
    return L

def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def computeQualityMeasures(lP, lT):
    """compute Dice and Hausdorff distance"""

    quality = dict()
    labelPred = lP
    labelTrue = lT
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()
    return quality


def evaluation(gtpath, predpath):
    """prediction evaluation"""

    prednames = file_name(predpath)
    gtnames = []
    for pname in prednames:
        gtnames.append(pname.split('_')[0] + '_segmentation.mhd')
    Hausdorff_distance = []
    dices = []
    for i in range(len(gtnames)):
        gt = sitk.ReadImage(os.path.join(gtpath, gtnames[i]))
        pred = sitk.ReadImage(os.path.join(predpath, prednames[i]))
        quality = computeQualityMeasures(pred, gt)
        Hausdorff_distance.append(quality['Hausdorff'])
        dices.append(quality['dice'])
    print('avg_Hausdorff_distance:', np.array(Hausdorff_distance).mean())
    print('avg_dice:', np.array(dices).mean())


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def dynamic_lr(config, lr, base_step):
    """dynamic learning rate generator"""
    base_lr = lr
    total_steps = int(base_step * config.epochs)
    warmup_steps = config.warmup_step
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * config.warmup_ratio))
        else:
            lr.append(a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))
    return lr
