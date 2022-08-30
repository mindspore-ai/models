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
"""dataprocess"""
import os
import pkuseg
from tqdm import tqdm

seg = pkuseg.pkuseg()


def changeListToText(content):
    """change list to text"""
    wordList = seg.cut(content)
    res = ""
    for item in wordList:
        res = res + ' ' + item
    return res


def changeIflytek(in_data_dir='', out_data_dir=''):
    """change iflytek"""
    changeList = ['dev.txt', 'train.txt', 'test.txt']
    for name in changeList:
        print(name)
        data = []
        with open(os.path.join(in_data_dir, name), 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                spData = line.split('_!_')
                content = spData[1].strip('\n').replace('\t', '')
                data.append({'content': content, 'label': spData[0]})
                line = f.readline()
        with open(os.path.join(out_data_dir, name), "w", encoding='utf-8') as f:
            for d in tqdm(data):
                content = changeListToText(d['content'])
                f.write(content + '\t' + d['label'] + '\n')


def changeTnews(in_data_dir='', out_data_dir=''):
    """change tnews"""
    changeDict = {'toutiao_category_dev.txt': 'dev.txt', 'toutiao_category_train.txt': 'train.txt',
                  'toutiao_category_test.txt': 'test.txt'}
    for k in changeDict:
        print(k)
        print(changeDict[k])
        data = []
        with open(os.path.join(in_data_dir, k), 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                spData = line.split('_!_')
                content = spData[3].strip('\n').replace('\t', '')
                data.append({'content': content, 'label': spData[1]})
                line = f.readline()
        with open(os.path.join(out_data_dir, changeDict[k]), "w", encoding='utf-8') as f:
            for d in tqdm(data):
                content = changeListToText(d['content'])
                f.write(content + '\t' + d['label'] + '\n')
