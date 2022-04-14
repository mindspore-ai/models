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
"""Evaluation"""

import os
import time
import argparse
import datetime
import numpy as np

from mindspore import Tensor, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size, release
from mindspore.common import dtype as mstype

from src.logging import get_logger
from src.models.resnest import get_network
from src.datasets.dataset import ImageNet
from src.config import config_train as config

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt_file in zip(top5_arg, gt_class):
        if gt_file in top5:
            sub_count += 1
    return sub_count

def get_result(args, model, top1_correct, top5_correct, img_tot):
    """calculate top1 and top5 value"""
    results = [[top1_correct], [top5_correct], [img_tot]]
    args.logger.info('before results={}'.format(results))
    if args.run_distribute:
        model_md5 = model.replace("/", "")
        tmp_dir = "/cache"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        top1_correct_npy = '/cache/top1_rank_{}_{}.npy'.format(args.rank, model_md5)
        top5_correct_npy = '/cache/top5_rank_{}_{}.npy'.format(args.rank, model_md5)
        img_tot_npy = '/cache/img_tot_rank_{}_{}.npy'.format(args.rank, model_md5)
        np.save(top1_correct_npy, top1_correct)
        np.save(top5_correct_npy, top5_correct)
        np.save(img_tot_npy, img_tot)
        while True:
            rank_ok = True
            for other_rank in range(args.group_size):
                top1_correct_npy = '/cache/top1_rank_{}_{}.npy'.format(other_rank, model_md5)
                top5_correct_npy = '/cache/top5_rank_{}_{}.npy'.format(other_rank, model_md5)
                img_tot_npy = '/cache/img_tot_rank_{}_{}.npy'.format(other_rank, model_md5)
                if not os.path.exists(top1_correct_npy) or not os.path.exists(top5_correct_npy) or \
                   not os.path.exists(img_tot_npy):
                    rank_ok = False
            if rank_ok:
                break

        top1_correct_all = 0
        top5_correct_all = 0
        img_tot_all = 0
        for other_rank in range(args.group_size):
            top1_correct_npy = '/cache/top1_rank_{}_{}.npy'.format(other_rank, model_md5)
            top5_correct_npy = '/cache/top5_rank_{}_{}.npy'.format(other_rank, model_md5)
            img_tot_npy = '/cache/img_tot_rank_{}_{}.npy'.format(other_rank, model_md5)
            top1_correct_all += np.load(top1_correct_npy)
            top5_correct_all += np.load(top5_correct_npy)
            img_tot_all += np.load(img_tot_npy)
        results = [[top1_correct_all], [top5_correct_all], [img_tot_all]]
        results = np.array(results)
    else:
        results = np.array(results)
    args.logger.info('after results={}'.format(results))
    return results

def Parse(args=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore resnest testing')
    # path arguments
    parser.add_argument('--outdir', type=str, default='output', help='logger output directory')
    parser.add_argument('--ckpt_path', type=str, default=".",
                        help='put the path to save checkpoint path')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')

    parser.add_argument('--pretrained_ckpt_path', type=str, default="./output/ckpt_0/resnest50-270_2502.ckpt",
                        help='put the path to resuming file if needed')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
    args = parser.parse_args()

    # init distributed
    if args.run_distribute:
        if args.device_target == "Ascend":
            init()
        elif args.device_target == "GPU":
            init("nccl")
        args.rank = get_rank()
        args.group_size = get_group_size()
    else:
        args.rank = 0
        args.group_size = 1

    args.outputs_dir = os.path.join(args.outdir,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    args.logger = get_logger(args.outputs_dir, args.rank)

    return args

def test():
    """test"""
    args = Parse()
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    else:
        context.set_context(device_id=0)

    if os.getenv('RANK_SIZE', "not_set").isdigit():
        device_num = int(os.getenv('RANK_SIZE'))
    else:
        device_num = 1

    # initialize distributed
    if args.run_distribute:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1

    # initialize logger
    outputs_dir = os.path.join(args.outdir, "valid",
                               datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(outputs_dir, config.rank)

    args.logger.save_args(args)

    # initialize dataset
    dataset = ImageNet(config.root, mode="valid",
                       img_size=config.base_size, crop_size=config.crop_size,
                       rank=config.rank, group_size=config.group_size, epoch=1,
                       batch_size=config.test_batch_size)
    eval_dataloader = dataset.create_tuple_iterator(output_numpy=True, num_epochs=1)

    # initialize model
    args.logger.important_info('start create network')
    net = get_network(config.net_name, True, args.pretrained_ckpt_path)

    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    if target == "Ascend":
        net.to_float(mstype.float16)
    net.set_train(False)
    t_end = time.time()
    it_name = 0
    for data, gt_classes in eval_dataloader:
        output = net(Tensor(data, mstype.float32))
        output = output.asnumpy()

        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[:, -5:]

        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += config.batch_size

        if config.rank == 0 and it_name == 0:
            t_end = time.time()
            it_name = 1

    if config.rank == 0:
        time_used = time.time() - t_end
        fps = (img_tot - config.batch_size) * config.group_size / time_used
        args.logger.info('Inference Performance: {:.2f} img/sec'.format(fps))

    results = get_result(args, args.pretrained_ckpt_path, top1_correct, top5_correct, img_tot)
    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    acc5 = 100.0 * top5_correct / img_tot
    args.logger.info('after allreduce eval: top1_correct={}, tot={},'
                     'acc={:.2f}%(TOP1)'.format(top1_correct, img_tot, acc1))
    args.logger.info('after allreduce eval: top5_correct={}, tot={},'
                     'acc={:.2f}%(TOP5)'.format(top5_correct, img_tot, acc5))

    if args.run_distribute:
        release()

if __name__ == "__main__":
    test()
