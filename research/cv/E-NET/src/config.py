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
"""argument config"""
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
parser.add_argument('--mindrecord_train_data', type=str)
parser.add_argument('--stage', type=int)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--num_class', type=int, default=20)
parser.add_argument('--device_target', type=str, default='Ascend')
config = parser.parse_args()

max_lr = config.lr
run_distribute = getBool(config.run_distribute)
global_size = int(os.environ["RANK_SIZE"])
repeat = config.repeat
stage = config.stage
ckpt_path = config.ckpt_path
save_path = config.save_path
device_target = config.device_target

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target=device_target)
if not run_distribute:
    context.set_context(device_id=int(os.environ["DEVICE_ID"]))
context.set_context(save_graphs=False)

seed_seed(2) # init random seed
weight_init = XavierUniform() # weight init
ms_train_data = config.mindrecord_train_data
num_class = config.num_class

# train config
class TrainConfig_1:
    """encoder training stage 1"""
    def __init__(self):
        self.subset = "train"
        self.num_class = 20
        self.train_img_size = 512
        self.epoch_num_save = 20
        self.epoch = 65
        self.encode = True
        self.attach_decoder = False
        self.lr = getLR(max_lr, 0, 150, 496, \
            run_distribute=run_distribute, global_size=global_size, repeat=repeat)

class TrainConfig_2:
    """encoder training stage 2"""
    def __init__(self):
        self.subset = "train"
        self.num_class = 20
        self.train_img_size = 512
        self.epoch_num_save = 20
        self.epoch = 85
        self.encode = True
        self.attach_decoder = False
        self.lr = getLR(max_lr, 65, 150, 496, \
            run_distribute=run_distribute, global_size=global_size, repeat=repeat)

class TrainConfig_3:
    """Enet training stage 3"""
    def __init__(self):
        self.subset = "train"
        self.num_class = 20
        self.train_img_size = 512
        self.epoch_num_save = 20
        self.epoch = 100
        self.encode = False
        self.attach_decoder = True
        self.lr = getLR(max_lr, 0, 100, 496, \
            run_distribute=run_distribute, global_size=global_size, repeat=repeat)
