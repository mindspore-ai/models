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
# matlab code:
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference: https://github.com/amdegroot/ssd.pytorch/blob/master/train.py

import os
import time
from random import shuffle

import numpy as np

from src.models.ADNet import adnet, WithLossCell_ADNET
from src.utils.get_train_videos import get_train_videos
from src.datasets.sl_dataset import initialize_pos_neg_dataset
from src.utils.augmentations import ADNet_Augmentation
from src.utils.get_wrapper_utils import get_dataLoader

from mindspore.communication.management import get_rank, get_group_size
from mindspore import nn, Tensor
from mindspore import ops
from mindspore import save_checkpoint
from mindspore.common.initializer import initializer
from mindspore import dtype as mstype
from mindspore.nn import TrainOneStepCell
import mindspore.numpy as nps


def adnet_train_sl(args, opts):

    train_videos = get_train_videos(opts, args)
    opts['num_videos'] = len(train_videos['video_names'])

    net, domain_specific_nets = adnet(opts=opts, trained_file=args.resume, multidomain=args.multidomain,
                                      distributed=args.distributed, run_online=args.run_online)

    optimizer = nn.SGD([{'params': net.base_network.trainable_params(), 'lr': 1e-4},
                        {'params': net.fc4_5.trainable_params()},
                        {'params': net.fc6.trainable_params()},
                        {'params': net.fc7.trainable_params()}],
                       learning_rate=1e-3,
                       momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])
    net.set_train()

    if not args.resume:
        print('Initializing weights...')
        scal = Tensor([0.01], mstype.float32)
        init_net(net, scal, args)

    criterion = ops.SparseSoftmaxCrossEntropyWithLogits()
    net_action_with_criterion = WithLossCell_ADNET(net, criterion, 'action')
    net_score_with_criterion = WithLossCell_ADNET(net, criterion, 'score')
    net_action = TrainOneStepCell(net_action_with_criterion, optimizer)
    net_score = TrainOneStepCell(net_score_with_criterion, optimizer)
    print('generating Supervised Learning dataset..')

    datasets_pos, datasets_neg = initialize_pos_neg_dataset(train_videos, opts, transform=ADNet_Augmentation(opts))
    number_domain = opts['num_videos']

    # calculating number of data
    len_dataset_pos = 0
    len_dataset_neg = 0
    for dataset_pos in datasets_pos:
        len_dataset_pos += len(dataset_pos)
    for dataset_neg in datasets_neg:
        len_dataset_neg += len(dataset_neg)

    epoch_size_pos = len_dataset_pos // opts['minibatch_size']
    epoch_size_neg = len_dataset_neg // opts['minibatch_size']
    if args.distributed:
        rank_id = get_rank()
        rank_size = get_group_size()
        epoch_size_pos = epoch_size_pos // rank_size
        epoch_size_neg = epoch_size_neg // rank_size
    else:
        rank_id = 0
    epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations

    print("1 epoch = " + str(epoch_size) + " iterations")

    max_iter = opts['numEpoch'] * epoch_size
    print("maximum iteration = " + str(max_iter))
    batch_iterators_pos, batch_iterators_neg = [], []
    dataloder_pos, dataloder_neg = get_dataLoader((datasets_pos, datasets_neg), opts, args,
                                                  ["im", "bbox", "action_label", "score_label", "vid_idx"])
    for data_pos in dataloder_pos:
        batch_iterators_pos.append(iter(data_pos))
    for data_neg in dataloder_neg:
        batch_iterators_neg.append(iter(data_neg))
    print('initial dataloader finished')
    epoch = args.start_epoch
    if epoch != 0 and args.start_iter == 0:
        start_iter = epoch * epoch_size
    else:
        start_iter = args.start_iter

    which_dataset = list(np.full(epoch_size_pos, fill_value=1))
    which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
    shuffle(which_dataset)

    which_domain = np.random.permutation(number_domain)

    # training loop
    print('start training')
    for iteration in range(start_iter, max_iter):
        if args.multidomain:
            curr_domain = which_domain[iteration % len(which_domain)]
        else:
            curr_domain = 0
        # if new epoch (not including the very first iteration)
        if (iteration != start_iter) and (iteration % epoch_size == 0):
            epoch += 1
            shuffle(which_dataset)
            np.random.shuffle(which_domain)

            if rank_id == 0:
                print('Saving state, epoch:', epoch)
                save_checkpoint(net, os.path.join(args.save_path, args.save_folder, args.save_file) +
                                'epoch' + repr(epoch) + '.ckpt')

                # save domain_specific

                for curr_domain, domain_specific_net in enumerate(domain_specific_nets):
                    save_checkpoint(domain_specific_net,
                                    os.path.join(args.save_path, args.save_folder, args.save_domain_dir,
                                                 'epoch' + repr(epoch) + '_' + str(curr_domain) + '.ckpt'))
        train(net, domain_specific_nets, curr_domain, which_dataset,
              batch_iterators_pos, batch_iterators_neg, dataloder_pos, dataloder_neg, iteration,
              net_action, net_score)
    # final save
    if rank_id == 0:
        save_checkpoint(net, os.path.join(args.save_path, args.save_folder, args.save_file) + 'final.ckpt')

        for curr_domain, domain_specific_net in enumerate(domain_specific_nets):
            save_checkpoint(domain_specific_net,
                            os.path.join(args.save_path, args.save_folder, args.save_domain_dir,
                                         'final' + '_' + str(curr_domain) + '.ckpt'))
    if args.run_online == 'True':
        import moxing as mox
        mox.file.copy_parallel('/cache/train_out/weights', args.train_url)
    return net, domain_specific_nets, train_videos


def train(net, domain_specific_nets, curr_domain, which_dataset, batch_iterators_pos, batch_iterators_neg,
          dataloder_pos, dataloder_neg, iteration, net_action, net_score):
    net.load_domain_specific(domain_specific_nets[curr_domain])
    # load train data
    flag_pos = which_dataset[iteration % len(which_dataset)]
    if flag_pos:  # if positive
        try:
            images, _, action_label, score_label, _ = next(batch_iterators_pos[curr_domain])
        except StopIteration:
            batch_iterators_pos[curr_domain] = iter(dataloder_pos[curr_domain])
            images, _, action_label, score_label, _ = next(batch_iterators_pos[curr_domain])
    else:
        try:
            images, _, action_label, score_label, _ = next(batch_iterators_neg[curr_domain])
        except StopIteration:
            batch_iterators_neg[curr_domain] = iter(dataloder_neg[curr_domain])
            images, _, action_label, score_label, _ = next(batch_iterators_neg[curr_domain])
    images = Tensor(images).transpose(0, 3, 1, 2)
    action_label = Tensor(action_label, dtype=mstype.float32)
    score_label = Tensor(score_label, dtype=mstype.int32)
    if flag_pos:
        action_l = net_action(images, ops.Argmax(1, output_type=mstype.int32)(action_label))
    else:
        action_l = Tensor([0])
    t0 = time.time()
    # load ADNetDomainSpecific with video index
    score_l = net_score(images, score_label)
    loss = action_l + score_l

    domain_specific_nets[curr_domain].load_weights_from_adnet(net)

    t1 = time.time()

    if iteration % 10 == 0:
        print('Timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.asnumpy()), end=' ')


def init_net(net, scal, args):
    if args.distributed:
        net.init_parameters_data(auto_parallel_mode=True)
    else:
        net.init_parameters_data(auto_parallel_mode=False)
    # fc 4
    net.fc4_5[0].weight.set_data(initializer('Normal', net.fc4_5[0].weight.shape, mstype.float32))
    net.fc4_5[0].weight.data.set_data(net.fc4_5[0].weight.data * scal.expand_as(net.fc4_5[0].weight.data))
    net.fc4_5[0].bias.set_data(nps.full(shape=net.fc4_5[0].bias.shape, fill_value=0.1))
    # fc 5
    net.fc4_5[3].weight.set_data(initializer('Normal', net.fc4_5[3].weight.shape, mstype.float32))
    net.fc4_5[3].weight.set_data(net.fc4_5[3].weight.data * scal.expand_as(net.fc4_5[3].weight.data))
    net.fc4_5[3].bias.set_data(nps.full(shape=net.fc4_5[3].bias.shape, fill_value=0.1))
    # fc 6
    net.fc6.weight.set_data(initializer('Normal', net.fc6.weight.shape, mstype.float32))
    net.fc6.weight.set_data(net.fc6.weight.data * scal.expand_as(net.fc6.weight.data))
    net.fc6.bias.set_data(nps.full(shape=net.fc6.bias.shape, fill_value=0))
    # fc 7
    net.fc7.weight.set_data(initializer('Normal', net.fc7.weight.shape, mstype.float32))
    net.fc7.weight.set_data(net.fc7.weight.data * scal.expand_as(net.fc7.weight.data))
    net.fc7.bias.set_data(nps.full(shape=net.fc7.bias.shape, fill_value=0))
