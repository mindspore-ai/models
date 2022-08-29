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
"""preprocess data"""
import argparse
import os
import shutil

from mindspore import context
import numpy as np
from src.config import Config
from src.dataset import build_dataset, build_dataloader

parser = argparse.ArgumentParser(description='HyperText Text Classification')
parser.add_argument('--model', type=str, default='HyperText',
                    help='HyperText')
parser.add_argument('--datasetdir', default='./data/iflytek_public', type=str,
                    help='dataset dir iflytek_public tnews_public')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--datasetType', default='iflytek', type=str, help='iflytek/tnews')
parser.add_argument('--device', default='Ascend', type=str, help='device GPU Ascend')
args = parser.parse_args()

config = Config(args.datasetdir, None, args.device)
if args.datasetType == 'tnews':
    config.useTnews()
else:
    config.useIflyek()
context.set_context(mode=context.GRAPH_MODE, device_target=config.device)
vocab, train_data, dev_data, test_data = build_dataset(config, use_word=True, min_freq=int(config.min_freq))
test_iter = build_dataloader(test_data, args.batch_size, config.max_length)
config.n_vocab = len(vocab)
ids_path = os.path.join('./preprocess_Result/', "00_ids")
ngrad_path = os.path.join('./preprocess_Result/', "01_ngrad")
label_path = os.path.join('./preprocess_Result/', "02_label")
if os.path.isdir(ids_path):
    shutil.rmtree(ids_path)
if os.path.isdir(ngrad_path):
    shutil.rmtree(ngrad_path)
if os.path.isdir(label_path):
    shutil.rmtree(label_path)

os.makedirs(ids_path)
os.makedirs(ngrad_path)
os.makedirs(label_path)

print('----------start test model-------------')
idx = 0
for d in test_iter.create_dict_iterator():
    ids_rst = d['ids'].asnumpy().astype(np.int32)
    ids_name = "hypertext_ids_bs" + str(args.batch_size) + "_" + str(idx) + ".bin"
    ids_real_path = os.path.join(ids_path, ids_name)
    ids_rst.tofile(ids_real_path)
    ngrad_ids = d['ngrad_ids'].asnumpy().astype(np.int32)
    ngrad_name = "hypertext_ngrad_bs" + str(args.batch_size) + "_" + str(idx) + ".bin"
    ngrad_real_path = os.path.join(ngrad_path, ngrad_name)
    ngrad_ids.tofile(ngrad_real_path)
    label = d['label'].asnumpy().astype(np.int32)
    label_name = "hypertext_label_bs" + str(args.batch_size) + "_" + str(idx) + ".bin"
    label_real_path = os.path.join(label_path, label_name)
    label.tofile(label_real_path)
    idx += 1
