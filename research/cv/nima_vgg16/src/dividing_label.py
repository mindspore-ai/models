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
"""Divide labels for train and test."""

import os
import random

import pandas as pd

from config import config


def divide_labels():
    """divide labels for train and test."""
    random.seed(10)

    if config.enable_modelarts:
        import moxing as mox
        mox.file.shift('os', 'mox')
    pic_names = os.listdir(config.data_path)
    dic = []
    with open(config.label_path) as f:
        for line in f:
            name = line.split()[1]+'.jpg'
            lst = map(int, line.split()[2:12])
            lst = list(lst)
            score = round(sum([(i+1)*j for i, j in enumerate(lst)])/sum(lst), 7)
            dic.append([name]+line.split()[2:12]+[score])
    df = pd.DataFrame(dic)
    df_new = df[df[0].isin(pic_names)].copy()
    df_new.reset_index(drop=True, inplace=True)
    test_img = random.sample(pic_names, 25597)

    test_label = df_new[df_new[0].isin(test_img)].copy()
    train_label = df_new[~df_new[0].isin(test_img)].copy()
    test_label.to_csv(config.val_label_path, header=0)
    train_label.to_csv(config.train_label_path, header=0)


if __name__ == '__main__':
    divide_labels()
