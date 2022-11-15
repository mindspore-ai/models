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
"""utils for train.py and eval.py"""
import numpy as np
from model.norm import USNorm


# evaluate on clean images with single norm
def evaluate_standard(test_loader, model, args):
    test_loss = 0
    test_acc = 0
    total_num = 0

    if args.dataset == 'cifar10':
        for _, (X, y) in enumerate(test_loader):
            output = model(X)
            # import pdb; pdb.set_trace()
            test_acc += (output.argmax(1) == y).sum().asnumpy()
            total_num += y.shape[0]
    elif args.dataset == 'cifar100':
        for _, (X, _, y) in enumerate(test_loader):
            output = model(X)
            # import pdb; pdb.set_trace()
            test_acc += (output.argmax(1) == y).sum().asnumpy()
            total_num += y.shape[0]

    return test_loss / total_num, test_acc / total_num


# evaluate on clean images with random norms
def evaluate_standard_random_norms(test_loader, model, args):
    test_loss = 0
    test_acc = 0
    total_num = 0

    if args.dataset == 'cifar10':
        for _, (X, y) in enumerate(test_loader):
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)
            output = model(X)
            # import pdb; pdb.set_trace()
            test_acc += (output.argmax(1) == y).sum().asnumpy()
            total_num += y.shape[0]
    elif args.dataset == 'cifar100':
        for _, (X, _, y) in enumerate(test_loader):
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)
            output = model(X)
            # import pdb; pdb.set_trace()
            test_acc += (output.argmax(1) == y).sum().asnumpy()
            total_num += y.shape[0]

    return test_loss / total_num, test_acc / total_num


# random norm for entire network
def set_random_norm(args, model):
    norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    norm = np.random.choice(norm_list)
    model.set_norms(norm)
    return norm


# random norm for each layer
def set_random_norm_mixed(args, model):
    norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    # get norm module from model
    for _, module in model.cells_and_names():
        if isinstance(module, USNorm):
            norm = np.random.choice(norm_list)
            module.set_norms(norm)


# setup random space for norms
def set_norm_list(min_group, max_group, random_type, gn_type):
    num_group_list = []
    for i in range(min_group, max_group + 1):
        num_group_list.append(2 ** i)
    # define norm list
    norm_list = []
    if 'bn' in random_type:
        norm_list.append('bn')
    if 'in' in random_type:
        norm_list.append('in')
    if '_' not in gn_type:
        for item in num_group_list:
            norm_list.append(gn_type + '_' + str(item))
    else:
        gn_str = gn_type[:gn_type.index('_')]
        gbn_str = gn_type[gn_type.index('_')+1:]
        for item in num_group_list:
            norm_list.append(gn_str + '_' + str(item))
            norm_list.append(gbn_str + '_' + str(item))

    return norm_list
