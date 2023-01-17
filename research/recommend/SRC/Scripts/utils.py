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
from mindspore import Tensor
from numpy import int32
from numpy.random import randint

from .Agent import SRC, MPC
from .Agent.utils import generate_path


def load_agent(args):
    if args.agent == 'SRC':
        return SRC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            weight_size=args.hidden_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            with_kt=args.withKT
        )
    if args.agent == 'MPC':
        return MPC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout,
            hor=args.hor
        )
    raise NotImplementedError


def get_data(batchSize, skillNum, target_num, initial_len, path_type, n):
    targets = Tensor(randint(0, skillNum, (batchSize, target_num), dtype=int32))
    initial_logs = Tensor(randint(0, skillNum, (batchSize, initial_len), dtype=int32))
    paths = Tensor(generate_path(batchSize, skillNum, path_type, n))
    return targets, initial_logs, paths
