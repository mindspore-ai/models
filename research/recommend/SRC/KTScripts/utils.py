# Copyright 2023 Huawei Technologies Co., Ltd
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
import random
from argparse import Namespace

import numpy as np
from mindspore import set_seed
from sklearn.metrics import roc_auc_score

from KTScripts.PredictModel import PredictModel, PredictRetrieval


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    print(f"random seed set to be {seed}")


def load_model(args: (Namespace, dict)):
    if isinstance(args, dict):
        args = Namespace(**args)
    if args.model in ('DKT', 'Transformer', 'GRU4Rec'):
        return PredictModel(
            feat_nums=args.feat_nums,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            output_size=args.output_size,
            with_label=not args.without_label,
            model_name=args.model)
    if args.model == 'CoKT':
        return PredictRetrieval(
            feat_nums=args.feat_nums,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            with_label=not args.without_label,
            model_name=args.model)
    raise NotImplementedError


def evaluate_utils(y_, y):
    if not isinstance(y_, np.ndarray):
        y_, y = y_.asnumpy(), y.asnumpy()
    acc = np.mean(np.equal(np.argmax(y_, -1) if len(y_.shape) > 1 else y_ > 0.5, y))
    auc = acc
    if not (np.equal(y, y[0])).all():
        if len(y_.shape) == 1:
            auc = roc_auc_score(y_true=y, y_score=y_)
    return acc, auc
