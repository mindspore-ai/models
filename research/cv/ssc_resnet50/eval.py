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
"""Evaluation for Network"""
import os
import logging
from glob import glob

import numpy as np
import mindspore.dataset as ds
from mindspore import dtype as mstype
from mindspore import context
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net
from mindspore import ops

from src.network.model import ModelBaseDis
from src.utils import AverageMeter, setup_default_logging
from src.dataset import CoMatchDatasetImageNetTest
from src.model_utils.config import get_config


def run_eval(args):
    logging.info("start create network!")
    netbase = ModelBaseDis(args, True)

    ckpt_list = []
    if args.folder:
        ckpt_list = sorted(glob(args.eval_pre_trained+"*.ckpt"),
                           key=lambda x: os.path.getmtime(os.path.join(args.pre_trained, x)))
    else:
        ckpt_list.append(args.eval_pre_trained)

    for pre_trained in ckpt_list:
        args.pre_trained = pre_trained
        logging.info("args.pre_trained exists, value: %s", args.pre_trained)
        param_dict = load_checkpoint(args.pre_trained)

        base_name = os.path.basename(args.pre_trained)
        epoch = int(base_name.split(".")[0].split("_")[1]) if base_name.startswith('epoch_') else -1

        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('comatch_network.encode.'):
                param_dict_new[key[16:]] = values

        para_not_list, _ = load_param_into_net(netbase, param_dict_new, strict_load=True)
        netbase.set_train(False)
        logging.info('param not load: %s', str(para_not_list))
        logging.info("load_checkpoint success!!")
        param_dict = None

        validate(args, netbase, epoch)


def validate(args, net, current_epoch):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    test_root = args.test_root
    test_dataset_generator = CoMatchDatasetImageNetTest(args, test_root)
    dataset = ds.GeneratorDataset(test_dataset_generator, ["label", "target"],
                                  shuffle=False, num_parallel_workers=8)

    dataset = dataset.batch(batch_size=32)
    steps_per_epoch = dataset.get_dataset_size()
    logging.info("validation steps_per_epoch %d", steps_per_epoch)
    data_loader = dataset.create_dict_iterator(num_epochs=1)
    for i, data in enumerate(data_loader):
        output, target, _, _ = net(data['label'], 1, 1, 1, data['target'], True)
        acc1, acc5 = accuracy_numpy(output.asnumpy(), target.asnumpy(), topk=(1, 5))
        top1.update(acc1)
        top5.update(acc5)
        if i % args.print_freq == 0:
            logging.info("validation || epoch: %d, iter: %5d. acc1 : %.5f. acc5 : %.5f.",
                         current_epoch, i + 1, float(top1.avg), float(top5.avg))
    logging.info("validation || epoch: %d, acc1 : %.5f. acc5 : %.5f.",
                 current_epoch, float(top1.avg), float(top5.avg))


def accuracy_numpy(output, target, topk=(1,)):
    maxk = np.max(topk)
    batch_size = target.shape[0]

    _, pred = topk_(output, maxk, 1)
    pred = np.transpose(pred)
    target = np.reshape(target, (1, -1))

    correct = np.equal(pred, target)

    res = []
    for k in topk:
        correct_k1 = np.reshape(correct[:k], (-1,))
        correct_k2 = np.sum(correct_k1, 0)
        x2 = (100 / batch_size)
        res.append(correct_k2 * x2)
    return res


def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = ops.TopK(sorted=True)(output, maxk)
    pred = pred.T
    target = ops.Reshape()(target, (1, -1))
    target_exp = target.expand_as(pred)
    correct = ops.Equal()(pred, target_exp)
    correct = ops.Cast()(correct, mstype.float32)

    res = []
    for k in topk:
        correct_k1 = ops.Reshape()(correct[:k], (-1,))
        correct_k2 = ops.ReduceSum(keep_dims=True)(correct_k1, 0)
        x2 = ops.ScalarToTensor()(100 / batch_size, mstype.float32).reshape(1)

        res.append(correct_k2*x2)
    return res


if __name__ == '__main__':
    args_config = get_config()
    if args_config.folder == 1:
        args_config.exp_dir = args_config.pre_trained

    logger = setup_default_logging(args_config)
    logger.info(args_config)

    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', args_config.device_id))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_config.device_target,
                        device_id=device_id)
    args_config.rank = 0
    args_config.device_num = 1
    run_eval(args_config)
