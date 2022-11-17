# Copyright (c) 2021. Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from argparse import ArgumentParser
from mindspore import context
from mindspore.common.initializer import XavierUniform
from src.util import getBool, seed_seed, getLR

parser = ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--run_distribute', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--repeat', type=int)
parser.add_argument('--data_path', type=str)
parser.add_argument('--num_class', type=int)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--epoch', type=int)

config = parser.parse_args()

max_lr = config.lr
run_distribute = getBool(config.run_distribute)
global_size = int(os.environ["RANK_SIZE"])
repeat = config.repeat
ckpt_path = config.ckpt_path
save_path = config.save_path

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")
context.set_context(device_id=0)
context.set_context(save_graphs=False)

seed_seed() # init random seed
weight_init = XavierUniform() # weight init
data_path = config.data_path
num_class = config.num_class

# train config
class TrainConfig:
    def __init__(self):
        self.train_img_size = 512
        self.epoch_num_save = 1

        self.epoch = config.epoch
        self.encode = False
        self.attach_decoder = False
        self.lr = getLR(max_lr, 0, self.epoch, 496, \
            run_distribute=run_distribute, global_size=global_size, repeat=repeat)
