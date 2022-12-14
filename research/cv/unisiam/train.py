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
from __future__ import print_function

import os
import sys
import time
import random
import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_transforms
from model.unisiam import UniSiam
from model.resnet import resnet10, resnet18, resnet34, resnet50
from dataset.miniImageNet import miniImageNet
from evaluate import evaluate_fewshot
from build_transform import build_transform
from util import AverageMeter, adjust_learning_rate


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_path', type=str, default=None, help='path for saving')
    parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet'], help='dataset')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # optimization setting
    parser.add_argument('--lr', type=float, default=0.3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--lrd_step', action='store_true', help='decay learning rate per step')

    # self-supervision setting
    parser.add_argument('--backbone', type=str,
                        default='resnet18', choices=['resnet10', 'resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--size', type=int, default=224, help='input size')
    parser.add_argument('--temp', type=float, default=2.0, help='temperature for loss function')
    parser.add_argument('--lamb', type=float, default=0.1, help='lambda for uniform loss')
    parser.add_argument('--dim_hidden', type=int, default=None, help='hidden dim. of projection')

    # few-shot evaluation setting
    parser.add_argument('--n_way', type=int, default=5, help='n_way')
    parser.add_argument('--n_query', type=int, default=15, help='n_query')
    parser.add_argument('--n_test_task', type=int, default=3000, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int, default=20, help='episode_batch_size')

    args = parser.parse_args()

    args.lr = args.lr * args.batch_size / 256
    if (args.save_path is not None) and (not os.path.isdir(args.save_path)): os.makedirs(args.save_path)
    args.split_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'split')

    return args


def build_train_loader(args, device_num=None, rank_id=None):
    train_transform = build_transform(args.size)

    if args.dataset == 'miniImageNet':
        train_dataset = miniImageNet(
            data_path=args.data_path,
            split_path=args.split_path,
            partition='train')
    else:
        raise ValueError(args.dataset)

    def copy_column(x, y):
        return x, x, y

    train_dataset = ds.GeneratorDataset(
        train_dataset, ["image", "label"], shuffle=True, num_parallel_workers=args.num_workers,
        num_shards=device_num, shard_id=rank_id)
    train_dataset = train_dataset.map(
        operations=copy_column, input_columns=["image", "label"], output_columns=["image1", "image2", "label"],
        column_order=["image1", "image2", "label"], num_parallel_workers=args.num_workers)
    train_dataset = train_dataset.map(operations=train_transform, input_columns=["image1"],
                                      num_parallel_workers=args.num_workers, python_multiprocessing=True)
    train_dataset = train_dataset.map(operations=train_transform, input_columns=["image2"],
                                      num_parallel_workers=args.num_workers, python_multiprocessing=True)
    train_dataset = train_dataset.batch(args.batch_size)

    return train_dataset


def build_fewshot_loader(args, mode='test', device_num=None, rank_id=None):

    assert mode in ['train', 'val', 'test']

    resize_dict = {160: 182, 224: 256, 288: 330, 320: 366, 384: 438}
    resize_size = resize_dict[args.size]
    print('Image Size: {}({})'.format(args.size, resize_size))

    test_transform = [
        c_transforms.Decode(),
        c_transforms.Resize(resize_size),
        c_transforms.CenterCrop(args.size),
        c_transforms.Normalize(mean=(0.485*255, 0.456*255, 0.406*255), std=(0.229*255, 0.224*255, 0.225*255)),
        c_transforms.HWC2CHW(),
        ]
    print('test_transform: ', test_transform)

    if args.dataset == 'miniImageNet':
        test_dataset = miniImageNet(
            data_path=args.data_path,
            split_path=args.split_path,
            partition=mode)
    else:
        raise ValueError(args.dataset)

    test_dataset = ds.GeneratorDataset(
        test_dataset, ["image", "label"], shuffle=False, num_parallel_workers=args.num_workers,
        num_shards=device_num, shard_id=rank_id)
    test_dataset = test_dataset.map(operations=test_transform, input_columns=["image"],
                                    num_parallel_workers=args.num_workers, python_multiprocessing=True)
    test_dataset = test_dataset.batch(args.batch_size)

    return test_dataset


def build_model(args):
    model_dict = {'resnet10': resnet10, 'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
    encoder = model_dict[args.backbone]()
    model = UniSiam(encoder=encoder, lamb=args.lamb, temp=args.temp, dim_hidden=args.dim_hidden)
    print(model)
    return model


class TrainOneStep(nn.Cell):
    def __init__(self, model, optimizer):
        super(TrainOneStep, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, images1, images2):

        def forward_fn(images1, images2):
            images = ops.Concat(axis=0)((images1, images2))
            loss, loss_pos, loss_neg, std = self.model(images)
            return loss, loss_pos, loss_neg, std

        loss, loss_pos, loss_neg, std = forward_fn(images1, images2)
        grads = self.grad(forward_fn, self.weights)(images1, images2)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss, loss_pos, loss_neg, std


def train_one_epoch(train_loader, model, epoch, args):
    """one epoch training"""

    model.set_train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_hist = AverageMeter()
    loss_pos_hist = AverageMeter()
    loss_neg_hist = AverageMeter()
    std_hist = AverageMeter()

    end = time.time()

    n_iter = train_loader.get_dataset_size()

    for idx, batchs in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = batchs[0].shape[0]
        if args.lrd_step: adjust_learning_rate(args, model.optimizer, idx*1.0/n_iter+epoch, args.epochs)

        loss, loss_pos, loss_neg, std = model(batchs[0], batchs[1])

        loss_hist.update(loss.asnumpy(), bsz)
        loss_pos_hist.update(loss_pos.asnumpy(), bsz)
        loss_neg_hist.update(loss_neg.asnumpy(), bsz)
        std_hist.update(std.asnumpy(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss_pos {lossp.val:.3f} ({lossp.avg:.3f})\t'
                  'loss_neg {lossn.val:.3f} ({lossn.avg:.3f})\t'
                  'std {std.val:.3f} ({std.avg:.3f})'.format(
                      epoch, idx + 1, train_loader.get_dataset_size(), batch_time=batch_time,
                      data_time=data_time, loss=loss_hist, lossp=loss_pos_hist, lossn=loss_neg_hist, std=std_hist))
            sys.stdout.flush()

    return loss_hist.avg


def main():
    args = parse_option()
    print("{}".format(args).replace(', ', ',\n'))

    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')

    train_loader = build_train_loader(args)
    test_loader = build_fewshot_loader(args, 'test')

    model = build_model(args)
    optimizer = nn.SGD(model.trainable_params(), learning_rate=args.lr, weight_decay=args.wd, momentum=0.9)
    train_model = TrainOneStep(model, optimizer)

    for epoch in range(args.epochs):

        if not args.lrd_step: adjust_learning_rate(args, optimizer, epoch+1, args.epochs)

        time1 = time.time()
        _ = train_one_epoch(train_loader, train_model, epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

    _ = evaluate_fewshot(model.encoder, test_loader, n_way=args.n_way, n_shots='1,5',
                         n_query=args.n_query, n_tasks=args.n_test_task, classifier='LR', power_norm=True)

    if args.save_path is not None:
        save_file = os.path.join(args.save_path, 'last.pth')
        ms.save_checkpoint(model, save_file)

if __name__ == '__main__':
    main()
