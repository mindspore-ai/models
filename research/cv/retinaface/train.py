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

import argparse
import datetime
import math
import os
import moxing as mox

import mindspore
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import cfg_res50, cfg_mobile025
from src.dataset import create_dataset
from src.loss import MultiBoxLoss
from src.lr_schedule import adjust_learning_rate, warmup_cosine_annealing_lr

code_dir = os.path.dirname(__file__)
work_dir = os.getcwd()
print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

parser = argparse.ArgumentParser(description='Retinaface Training Args')
parser.add_argument("--modelarts_FLAG", type=bool, default=True, help="use modelarts or not")
parser.add_argument('--data_url', type=str, default='./data/')
parser.add_argument("--train_url", type=str, default="./output/checkpoint")
parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset/")
parser.add_argument("--training_dataset", type=str, default="/cache/dataset/label.txt")
parser.add_argument("--modelarts_result_dir", type=str, default="/cache/train_output/")
parser.add_argument('--backbone_name', type=str, default='ResNet50', help='backbone name')
parser.add_argument("--pretrain_path", type=str, default="/cache/dataset/retinaface_resnet50.ckpt")
parser.add_argument("--epoch", type=int, default=1, help="modelarts test epoch")

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
    According to the switch FLAGS, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=FLAGS.modelarts_result_dir, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".
          format(FLAGS.modelarts_result_dir, FLAGS.train_url))


def train_with_resnet(cfg, FLAGS):
    """train_with_resnet"""
    if FLAGS.modelarts_FLAG:
        obs_data2modelarts(FLAGS=FLAGS)
    mindspore.common.seed.set_seed(cfg['seed'])
    from src.network_with_resnet import RetinaFace, RetinaFaceWithLossCell, TrainingWrapper, resnet50
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    device_num = cfg['nnpu']
    rank = 0
    if cfg['device_target'] == "Ascend":
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
        else:
            context.set_context(device_id=cfg['device_id'])
    elif cfg['device_target'] == "GPU":
        if cfg['ngpu'] > 1:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()

    batch_size = cfg['batch_size']
    if FLAGS.modelarts_FLAG:
        cfg['epoch'] = FLAGS.epoch
    max_epoch = cfg['epoch']
    momentum = cfg['momentum']
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    loss_scale = cfg['loss_scale']
    initial_lr = cfg['initial_lr']
    gamma = cfg['gamma']
    T_max = cfg['T_max']
    eta_min = cfg['eta_min']
    if FLAGS.modelarts_FLAG:
        cfg['training_dataset'] = FLAGS.training_dataset
    training_dataset = cfg['training_dataset']
    num_classes = 2
    negative_ratio = 7
    stepvalues = (cfg['decay1'], cfg['decay2'])

    ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio, cfg['batch_size'])
    backbone = resnet50(1001)
    backbone.set_train(True)

    if cfg['pretrain'] and cfg['resume_net'] is None:
        if FLAGS.modelarts_FLAG:
            cfg['pretrain_path'] = FLAGS.pretrain_path
        pretrained_res50 = cfg['pretrain_path']
        param_dict_res50 = load_checkpoint(pretrained_res50)
        load_param_into_net(backbone, param_dict_res50)
        print('Load resnet50 from [{}] done.'.format(pretrained_res50))

    net = RetinaFace(phase='train', backbone=backbone)
    net.set_train(True)

    if cfg['resume_net'] is not None:
        pretrain_model_path = cfg['resume_net']
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print('Resume Model from [{}] Done.'.format(cfg['resume_net']))

    net = RetinaFaceWithLossCell(net, multibox_loss, cfg)

    if lr_type == 'dynamic_lr':
        lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                                  warmup_epoch=cfg['warmup_epoch'], lr_type1=lr_type)
    elif lr_type == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(initial_lr, steps_per_epoch, cfg['warmup_epoch'], max_epoch, T_max, eta_min)

    if cfg['optim'] == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, momentum, weight_decay, loss_scale)
    elif cfg['optim'] == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                               weight_decay=weight_decay, loss_scale=loss_scale)
    else:
        raise ValueError('optim is not define.')

    net = TrainingWrapper(net, opt)

    model = Model(net)

    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * 1,
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
    if FLAGS.modelarts_FLAG:
        if not os.path.exists(FLAGS.modelarts_result_dir):
            os.makedirs(FLAGS.modelarts_result_dir)
        cfg['ckpt_path'] = FLAGS.modelarts_result_dir + "ckpt_" + str(rank) + "/"
    else:
        cfg['ckpt_path'] = cfg['ckpt_path'] + "ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list)

    if FLAGS.modelarts_FLAG:
        modelarts_result2obs(FLAGS=FLAGS)


def train_with_mobilenet(cfg, FLAGS):
    """train_with_mobilenet"""
    if FLAGS.modelarts_FLAG:
        obs_data2modelarts(FLAGS=FLAGS)
    mindspore.common.seed.set_seed(cfg['seed'])
    from src.network_with_mobilenet import RetinaFace, RetinaFaceWithLossCell, TrainingWrapper, resnet50, mobilenet025
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)
    if context.get_context("device_target") == "GPU":
        # Enable graph kernel
        context.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")
    rank = 0
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
    if FLAGS.modelarts_FLAG:
        cfg['training_dataset'] = FLAGS.training_dataset
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
    if FLAGS.modelarts_FLAG:
        if not os.path.exists("cache/train_output/"):
            os.makedirs("cache/train_output/")
        cfg['ckpt_path'] = FLAGS.modelarts_result_dir + "ckpt_" + str(rank) + "/"
    else:
        cfg['ckpt_path'] = cfg['ckpt_path'] + "ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list, dataset_sink_mode=True)

    if FLAGS.modelarts_FLAG:
        modelarts_result2obs(FLAGS=FLAGS)


if __name__ == '__main__':
    if args_opt.backbone_name == 'ResNet50':
        config = cfg_res50
        train_with_resnet(cfg=config, FLAGS=args_opt)

    elif args_opt.backbone_name == 'MobileNet025':
        config = cfg_mobile025
        train_with_mobilenet(cfg=config, FLAGS=args_opt)

    print('train config:\n', config)
