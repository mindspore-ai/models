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
"""Train"""
import os
import time
import logging

from mindspore.nn import SGD
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore import save_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import Parameter

from src.network.model import ModelBaseDis, ModelWithLossCellDis, TrainOneStepWithLossScaleCellDist
from src.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
from src.utils import setup_default_logging, AverageMeter
from src.dataset import create_comatch_dataset
from src.model_utils.local_adapter import get_device_id, get_device_num, get_rank_id
from src.model_utils.config import get_config


def run_train(args):
    if args.rank == 0:
        logging.info("start create dataset!")
    dataset, data_size = create_comatch_dataset(args)
    args.steps_per_epoch = int(data_size / args.batch_size / args.device_num)
    if args.rank == 0:
        logging.info("step per epoch: %d", args.steps_per_epoch)

        # create the base network
        logging.info("start create network!")
    netbase = ModelBaseDis(args)

    # create the loss network
    netloss = ModelWithLossCellDis(args, netbase)

    # declare lr and optimizer
    lr = warmup_cosine_annealing_lr(args.lr, args.steps_per_epoch, args.warm_epochs, args.epochs, args.epochs, 0.)
    opt = SGD(params=netbase.trainable_params(), learning_rate=lr, momentum=args.momentum,
              weight_decay=args.weight_decay, loss_scale=1)

    # Dynamic loss scale.
    scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)

    # train use TrainOneStepWithLosssScaleCell
    net = TrainOneStepWithLossScaleCellDist(network=netloss, optimizer=opt,
                                            scale_update_cell=scale_manager.get_update_cell())
    net.set_train()
    if args.rank == 0:
        logging.info('finish create network')

    if args.pre_trained != "":
        if args.rank == 0:
            logging.info("args.pre_trained exists, value: %s", args.pre_trained)
        param_dict = load_checkpoint(args.pre_trained)
        base_name = os.path.basename(args.pre_trained)
        if args.rank == 0:
            logging.info("base name: %s", base_name)
        args.start_epoch = int(base_name.split(".")[0].split("_")[1]) + 1 if base_name.startswith('epoch_') else 0
        param_dict['learning_rate'] = Parameter(Tensor(lr, mstype.float32))
        param_dict['global_step'] = Parameter(Tensor([args.start_epoch * args.steps_per_epoch], mstype.int32))

        para_not_list, _ = load_param_into_net(net, param_dict, strict_load=True)
        if args.rank == 0:
            logging.info('param not load: %s', str(para_not_list))
            logging.info("load_checkpoint success!!")
        param_dict = None

    # create folder needs
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir, exist_ok=True)

    loss_x_meter = AverageMeter("loss_x")
    loss_u_meter = AverageMeter("loss_u")
    loss_contrast_meter = AverageMeter("loss_c")
    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    data_loader = dataset.create_dict_iterator(num_epochs=int(args.epochs - args.start_epoch))

    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        for i, data in enumerate(data_loader):
            data_time.update(time.time() - end)

            _, loss_x, loss_u, loss_contrast, cond, scaling_sens = net(data["label"], data['unlabel_weak'],
                                                                       data['unlabel_strong0'],
                                                                       data['unlabel_strong1'], data['target'])
            loss_x_meter.update(loss_x.asnumpy())
            loss_u_meter.update(loss_u.asnumpy())
            loss_contrast_meter.update(loss_contrast.asnumpy())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.rank == 0:
                logging.info("rank: %d, epoch: %3d, step: %4d, loss_x: %.3f, loss_u: %.3f, loss_contrast: %.3f, "
                             "batch_time: %.2f, data_time: %.2f, lr: %.6f, overflow: %.5f, scaling_sens: %d.",
                             args.rank, epoch, i, loss_x_meter.avg, loss_u_meter.avg, loss_contrast_meter.avg,
                             batch_time.avg, data_time.avg, lr[i + epoch * args.steps_per_epoch],
                             float(cond.asnumpy()), int(scaling_sens.asnumpy()))

        if args.rank == 0:
            append_dict = {}
            append_dict['epoch'] = epoch
            ckpt_file_name = os.path.join(args.exp_dir, "epoch_" + str(epoch) + ".ckpt")
            save_checkpoint(net, ckpt_file_name, append_dict=append_dict)
            save_checkpoint(net, os.path.join(args.exp_dir, 'model_last.ckpt'))
            if args.rank == 0:
                logging.info("ckpt generated from epoch %d saved.", epoch)
            loss_x_meter.reset()
            loss_u_meter.reset()
            loss_contrast_meter.reset()
            batch_time.reset()
            data_time.reset()


if __name__ == "__main__":
    args_config = get_config()
    set_seed(1)

    logger = setup_default_logging(args_config)
    logger.info(args_config)

    logger.info("start init dist!")
    context.set_context(mode=context.GRAPH_MODE, device_target=args_config.device_target,
                        device_id=int(get_device_id()))

    # set max_call_depth for dealing the Eval] Exceed function call depth limit 1000
    context.set_context(max_call_depth=4000)
    logger.info('is_distributed: %s', args_config.is_distributed)
    if args_config.is_distributed:
        if args_config.device_target == "Ascend":
            init("hccl")
            args_config.device_num = get_device_num()
            args_config.rank = get_rank_id()
            logger.info("init ascend dist rank: %d, device_num: %d.", args_config.rank, args_config.device_num)
            context.set_auto_parallel_context(device_num=args_config.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

        else:
            init("nccl")
            context.reset_auto_parallel_context()
            args_config.device_num = get_group_size()
            args_config.rank = get_rank()
            logger.info("init gpu dist rank: %d, device_num: %d.", args_config.rank, args_config.device_num)
            context.set_auto_parallel_context(device_num=args_config.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        raise ValueError('ResnetSSOD only support distributed now.')

    run_train(args_config)
