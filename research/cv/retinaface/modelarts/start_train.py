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
"""Train Retinaface_resnet50ormobilenet0.25."""

import os
import glob
import math
import argparse
import datetime
import moxing as mox
import numpy as np

import mindspore
from mindspore import context, Tensor, export
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.network_with_resnet import RetinaFace, RetinaFaceWithLossCell, TrainingWrapper, resnet50
from src.config import cfg_res50, cfg_mobile025
from src.loss import MultiBoxLoss
from src.dataset import create_dataset
from src.lr_schedule import adjust_learning_rate, warmup_cosine_annealing_lr

code_dir = os.path.dirname(__file__)
work_dir = os.getcwd()
print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

parser = argparse.ArgumentParser(description='train')

parser.add_argument("--train_url", type=str, default="./output/checkpoint")
parser.add_argument("--data_url", type=str, default="./data/train/")
parser.add_argument("--training_dataset", type=str, default="/cache/dataset/label.txt")
parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
parser.add_argument("--ckpt_path", type=str, default="/cache/result")
parser.add_argument("--pretrain_path", type=str, default="/cache/dataset/retinaface_resnet50.ckpt")

parser.add_argument('--backbone_name', type=str, default="ResNet50", help="backbone name")
parser.add_argument('--device_target', type=str, default="Ascend", help="device name")
parser.add_argument("--lr_type", type=str, default="dynamic_lr", help="dynamic_lr or cosine_annealing")
parser.add_argument('--optim', type=str, default="sgd", help="sgd or momentum'")
parser.add_argument("--pretrain", type=bool, default=False)
parser.add_argument("--clip", type=bool, default=False)
parser.add_argument("--resume_net", default=None)

parser.add_argument("--num_classes", type=int, default=2, help="transfer learning way2")
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--num_anchor", type=int, default=29126,
                    help="when transfer learning, this argument should be changed too")
parser.add_argument("--negative_ratio", type=int, default=7)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--nnpu', type=int, default=1)
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--warmup_epoch", type=int, default=-1, help="dynamic_lr: -1, cosine_annealing:0")
parser.add_argument("--image_size", type=int, default=840, help="transfer learning way1")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--initial_lr", type=float, default=0.04)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--T_max", type=int, default=50, help="when choosing cosine_annealing")
parser.add_argument("--eta_min", type=float, default=0.0, help="when choosing cosine_annealing")
parser.add_argument("--loss_scale", type=int, default=1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--decay1", type=int, default=20)
parser.add_argument("--decay2", type=int, default=40)
parser.add_argument("--stepvalues", type=list, default=[20, 40])
parser.add_argument("--keep_checkpoint_max", type=int, default=8)
parser.add_argument("--match_thresh", type=float, default=0.35)
parser.add_argument("--variance", type=list, default=[0.1, 0.2])
parser.add_argument("--loc_weight", type=float, default=2.0)
parser.add_argument("--class_weight", type=float, default=1.0)
parser.add_argument("--landm_weight", type=float, default=1.0)

args_opt = parser.parse_args()


def obs_data2modelarts(FLAGS):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(FLAGS.data_url, FLAGS.modelarts_data_dir))
    mox.file.copy_parallel(src_url=FLAGS.data_url, dst_url=FLAGS.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(FLAGS.modelarts_data_dir)
    print("===>>>Files:", files)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=FLAGS.modelarts_result_dir, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.modelarts_result_dir,
                                                                                  FLAGS.train_url))
    files = os.listdir()
    print("===>>>current Files:", files)
    mox.file.copy(src_url='Retinaface.air', dst_url=FLAGS.train_url+'/Retinaface.air')


def export_AIR(cfg):
    """start modelarts export"""
    ckpt_list = glob.glob(cfg.modelarts_result_dir + "*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=cfg.device_id)

    net_ = resnet50(1001)
    param_dict_ = load_checkpoint(ckpt_model)
    load_param_into_net(net_, param_dict_)

    input_arr = Tensor(np.zeros([1, 3, 5568, 1056], np.float32))
    export(net_, input_arr, file_name='Retinaface', file_format='AIR')


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def train_with_resnet(cfg):
    """train_with_resnet"""
    mindspore.common.seed.set_seed(cfg.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    device_num = cfg.nnpu
    rank = 0
    if cfg.device_target == "Ascend":
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
        else:
            context.set_context(device_id=cfg.device_id)
    elif cfg.device_target == "GPU":
        if cfg['ngpu'] > 1:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()

    ds_train = create_dataset(cfg.training_dataset, cfg_res50, cfg.batch_size,
                              multiprocessing=True, num_worker=cfg.num_workers)
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(cfg.num_classes, cfg.num_anchor, cfg.negative_ratio, cfg.batch_size)
    backbone = resnet50(1001)
    backbone.set_train(True)

    if cfg.pretrain and cfg.resume_net is None:
        pretrained_res50 = cfg.pretrain_path
        param_dict_res50 = load_checkpoint(pretrained_res50)
        filter_list = [x.name for x in backbone.end_point.get_parameters()]
        filter_checkpoint_parameter_by_list(param_dict_res50, filter_list)
        load_param_into_net(backbone, param_dict_res50)
        print('Load resnet50 from [{}] done.'.format(pretrained_res50))

    net = RetinaFace(phase='train', backbone=backbone)
    net.set_train(True)

    if cfg.resume_net is not None:
        pretrain_model_path = cfg.resume_net
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print('Resume Model from [{}] Done.'.format(cfg.resume_net))

    net = RetinaFaceWithLossCell(net, multibox_loss, cfg_res50)

    if cfg.lr_type == 'dynamic_lr':
        lr = adjust_learning_rate(cfg.initial_lr, cfg.gamma, cfg.stepvalues, steps_per_epoch,
                                  cfg.epoch, warmup_epoch=cfg.warmup_epoch, lr_type1=cfg.lr_type)
    elif cfg.lr_type == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(cfg.initial_lr, steps_per_epoch, cfg.warmup_epoch,
                                        cfg.epoch, cfg.T_max, cfg.eta_min)

    if cfg.optim == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, cfg.momentum, cfg.weight_decay, cfg.loss_scale)
    elif cfg.optim == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay, loss_scale=cfg.loss_scale)
    else:
        raise ValueError('optim is not define.')

    net = TrainingWrapper(net, opt)
    model = Model(net)
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * 1,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)

    cfg.ckpt_path = cfg.ckpt_path + "ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg.ckpt_path, config=config_ck)
    cfg.modelarts_result_dir = cfg.modelarts_result_dir + "ckpt_" + str(rank) + "/"

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(cfg.epoch, ds_train, callbacks=callback_list)


def train_with_mobilenet(cfg):
    """train_with_mobilenet"""
    mindspore.common.seed.set_seed(cfg['seed'])
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)
    if context.get_context("device_target") == "GPU":
        # Enable graph kernel
        context.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")
    if cfg['ngpu'] > 1:
        init("nccl")
        context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        cfg['ckpt_path'] = cfg['ckpt_path'] + "ckpt_" + str(get_rank()) + "/"

    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']

    momentum = cfg['momentum']
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    initial_lr = cfg['initial_lr']
    gamma = cfg['gamma']
    training_dataset = cfg['training_dataset']
    num_classes = 2
    negative_ratio = 7
    stepvalues = (cfg['decay1'], cfg['decay2'])

    ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio, cfg['batch_size'])
    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    backbone.set_train(True)

    if cfg['name'] == 'ResNet50' and cfg['pretrain'] and cfg['resume_net'] is None:
        pretrained_res50 = cfg['pretrain_path']
        param_dict_res50 = load_checkpoint(pretrained_res50)
        load_param_into_net(backbone, param_dict_res50)
        print('Load resnet50 from [{}] done.'.format(pretrained_res50))
    elif cfg['name'] == 'MobileNet025' and cfg['pretrain'] and cfg['resume_net'] is None:
        pretrained_mobile025 = cfg['pretrain_path']
        param_dict_mobile025 = load_checkpoint(pretrained_mobile025)
        load_param_into_net(backbone, param_dict_mobile025)
        print('Load mobilenet0.25 from [{}] done.'.format(pretrained_mobile025))

    net = RetinaFace(phase='train', backbone=backbone, cfg=cfg)
    net.set_train(True)

    if cfg['resume_net'] is not None:
        pretrain_model_path = cfg['resume_net']
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print('Resume Model from [{}] Done.'.format(cfg['resume_net']))

    net = RetinaFaceWithLossCell(net, multibox_loss, cfg)

    lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                              warmup_epoch=cfg['warmup_epoch'], lr_type1=lr_type)

    if cfg['optim'] == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, momentum)
    elif cfg['optim'] == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                               weight_decay=weight_decay, loss_scale=1)
    else:
        raise ValueError('optim is not define.')

    net = TrainingWrapper(net, opt)

    model = Model(net)

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list, dataset_sink_mode=True)


if __name__ == '__main__':

    if args_opt.backbone_name == 'ResNet50':

        obs_data2modelarts(args_opt)
        print('train config:\n', args_opt)
        train_with_resnet(args_opt)
        export_AIR(args_opt)
        modelarts_result2obs(args_opt)

    elif args_opt.backbone_name == 'MobileNet025':
        config = cfg_mobile025
        train_with_mobilenet(cfg=config)
