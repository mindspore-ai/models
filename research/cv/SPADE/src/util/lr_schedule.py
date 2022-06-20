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
""" custom lr schedule """

def dynamic_lr(opt, steps_per_epoch, begin_lr):
    """dynamic learning rate generator"""
    lr_each_peoch = []
    lr_each_step = []
    flag = True
    total_epoch = opt.total_epoch - opt.now_epoch
    epoch_num = opt.total_epoch - opt.decay_epoch
    for i in range(total_epoch):
        if i < opt.decay_epoch - opt.now_epoch:
            lr_each_peoch.append(begin_lr)
        else:
            if opt.decay_epoch - opt.now_epoch <= 0 and flag:
                flag = False
                lr = begin_lr - (opt.now_epoch - opt.decay_epoch) * begin_lr / epoch_num
            else:
                lr = lr_each_peoch[i-1]-begin_lr/epoch_num
            lr_each_peoch.append(lr)
    for i in range(total_epoch):
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_each_peoch[i])
    return lr_each_step
