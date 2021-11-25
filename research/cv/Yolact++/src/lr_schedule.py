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

"""lr generator for yolact"""

def dynamic_lr(cfg, start_epoch=0, total_epochs=0, steps_each_epoch=0):

    """dynamic learning rate generator"""
    start_steps = start_epoch * steps_each_epoch
    total_steps = (total_epochs - start_epoch) * steps_each_epoch
    lr_init = cfg['lr']
    gamma = cfg['gamma']


    lr_steps = (280000, 600000, 700000, 750000)
    lr_stages = [lr_init * (gamma ** 0),
                 lr_init * (gamma ** 1),
                 lr_init * (gamma ** 2),
                 lr_init * (gamma ** 3),
                 lr_init * (gamma ** 4)
                 ]

    lr = []
    lr_warmup_until = cfg['lr_warmup_until']
    lr_warmup_init = cfg['lr_warmup_init']

    step_index = 0
    # <=500 change
    # >500 <=280k lr_init
    # >280k <=600k lr_init * (gamma ** 1)
    # >600k <=700k lr_init * (gamma ** 2)
    # >700k <=750k lr_init * (gamma ** 3)
    # >750k        lr_init * (gamma ** 4)
    for cur_step in range(total_steps):
        # If cur_step reaches another stage, change the learning rate
        if lr_warmup_until > 0 and cur_step <= lr_warmup_until:
            lr.append((lr_init - lr_warmup_init) * (cur_step / lr_warmup_until) + lr_warmup_init)
        else:
            lr.append(lr_stages[step_index])
        # lr_init * (gamma ** 4)
        while step_index < len(lr_steps) and cur_step >= lr_steps[step_index]:
            step_index += 1

    learning_rate = lr[start_steps:]
    return learning_rate
