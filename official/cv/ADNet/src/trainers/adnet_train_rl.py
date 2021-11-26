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
# matlab code: https://github.com/hellbell/ADNet/blob/master/train/adnet_train_RL.m
# policy gradient in pytorch: https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
import os
import time
import copy

import numpy as np

from src.trainers.RL_tools import TrackingPolicyLoss
from src.datasets.rl_dataset import RLDataset
from src.models.CustomizedCell import WithLossCell, TrainOneStepCell
from src.utils.save_ckpt import save_ckpt
from src.utils.get_wrapper_utils import get_dataLoader

from mindspore import nn, ops
from mindspore.communication.management import get_rank, get_group_size


def adnet_train_rl(net, domain_specific_nets, train_videos, opts, args):
    if args.run_online == 'True':
        save_path = '/cache/train_out'
    else:
        save_path = ''
    if not os.path.exists(os.path.join(save_path, args.save_folder, args.save_domain_dir)):
        os.makedirs(os.path.join(save_path, args.save_folder, args.save_domain_dir))

    net.set_phase('test')

    optimizer = nn.SGD([{'params': net.base_network.trainable_params(), 'lr': 1e-4},
                        {'params': net.fc4_5.trainable_params()},
                        {'params': net.fc6.trainable_params()},
                        {'params': net.fc7.trainable_params(), 'lr': 0}],
                       learning_rate=1e-3, momentum=opts['train']['momentum'],
                       weight_decay=opts['train']['weightDecay'])
    criterion = TrackingPolicyLoss()
    clip_idx_epoch = 0
    prev_net = copy.deepcopy(net)
    dataset = RLDataset(prev_net, domain_specific_nets, train_videos, opts, args)
    rlnet_with_criterion = WithLossCell(net, criterion)
    net_rl = TrainOneStepCell(rlnet_with_criterion, optimizer)
    for epoch in range(args.start_epoch, opts['numEpoch']):
        if epoch != args.start_epoch:
            dataset.reset(prev_net, domain_specific_nets, train_videos, opts, args)
        data_loader = get_dataLoader(dataset, opts, args,
                                     ["log_probs_list", "reward_list", "vid_idx_list", 'patch'])
        # create batch iterator
        batch_iterator = iter(data_loader)

        epoch_size = len(dataset) // opts['minibatch_size']   # 1 epoch, how many iterations
        if args.distributed:
            rank_id = get_rank()
            rank_size = get_group_size()
            epoch_size = epoch_size // rank_size

        for iteration in range(epoch_size):
            # load train data
            # action, action_prob, log_probs, reward, patch, action_dynamic, result_box = next(batch_iterator)
            _, reward, vid_idx, patch = next(batch_iterator)

            # train
            tic = time.time()
            patch = patch.transpose(0, 3, 1, 2)
            # find out the unique value in vid_idx
            # separate the batch with each video idx
            if args.multidomain:
                vid_idx_unique = ops.Unique()(vid_idx)[0]
                for i in range(len(vid_idx_unique)):
                    choice_list = (vid_idx_unique[i] == vid_idx).asnumpy().nonzero()[0].tolist()
                    if len(choice_list) == 1:
                        continue
                    tmp_patch = patch[choice_list]
                    tmp_reward = reward[choice_list]
                    net_rl(tmp_patch, tmp_reward)
                    # save the ADNetDomainSpecific back to their module
                    idx = np.asscalar(vid_idx_unique[i].asnumpy())
                    domain_specific_nets[idx].load_weights_from_adnet(net)
            else:
                net_rl(patch, reward)

            toc = time.time() - tic
            print('epoch ' + str(epoch) + ' - iteration ' + str(iteration) + ' - train time: ' + str(toc) + " s")

            if iteration % 1000 == 0:
                if not args.distributed or rank_id == 0:
                    save_ckpt(net, domain_specific_nets, save_path, args, iteration, epoch, 1)

            clip_idx_epoch += 1

        if not args.distributed or rank_id == 0:
            save_ckpt(net, domain_specific_nets, save_path, args, iteration, epoch, 2)

    if not args.distributed or rank_id == 0:
        save_ckpt(net, domain_specific_nets, save_path, args, iteration, epoch, 3)

    if args.run_online == 'True':
        import moxing
        moxing.file.copy_parallel('/cache/train_out/weights', args.train_url)
    return net
