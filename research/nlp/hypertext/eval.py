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
"""eval file"""
import argparse

from mindspore import load_checkpoint, load_param_into_net, context
from mindspore.ops import Squeeze, Argmax
from mindspore.common import dtype as mstype
from mindspore import numpy as mnp
from src.config import Config
from src.dataset import build_dataset, build_dataloader
from src.hypertext import HModel

parser = argparse.ArgumentParser(description='HyperText Text Classification')
parser.add_argument('--model', type=str, default='HyperText',
                    help='HyperText')
parser.add_argument('--modelPath', default='./output/hypertext_iflytek.ckpt', type=str, help='save model path')
parser.add_argument('--datasetdir', default='./data/iflytek_public', type=str,
                    help='dataset dir iflytek_public tnews_public')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--datasetType', default='iflytek', type=str, help='iflytek/tnews')
parser.add_argument('--device', default='GPU', type=str, help='device GPU Ascend')
args = parser.parse_args()

config = Config(args.datasetdir, None, args.device)
if args.datasetType == 'tnews':
    config.useTnews()
else:
    config.useIflyek()
context.set_context(mode=context.GRAPH_MODE, device_target=config.device)
vocab, train_data, dev_data, test_data = build_dataset(config, use_word=True, min_freq=int(config.min_freq))
test_iter = build_dataloader(test_data, config.batch_size, config.max_length)
config.n_vocab = len(vocab)
model_path = args.modelPath
hmodel = HModel(config).to_float(mstype.float16)
param_dict = load_checkpoint(model_path)
load_param_into_net(hmodel, param_dict)
squ = Squeeze(-1)
argmax = Argmax(output_type=mstype.int32)
cur, total = 0, 0
print('----------start test model-------------')
for d in test_iter.create_dict_iterator():
    hmodel.set_train(False)
    out = hmodel(d['ids'], d['ngrad_ids'])
    predict = argmax(out)
    acc = predict == squ(d['label'])
    acc = mnp.array(acc, dtype=mnp.float32)
    cur += (mnp.sum(acc, -1))
    total += len(acc)
print('acc:', cur / total)
