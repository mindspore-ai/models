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

import argparse
import os
import time
from mindspore import context, nn
from mindspore.common import set_seed
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.communication import management as D
from mindspore.communication.management import get_group_size, get_rank
from src.pyramidbox import build_net, NetWithLoss, EvalLoss
from src.dataset import create_val_dataset, create_train_dataset
from src.config import cfg

MIN_LOSS = 10000

def parse_args():
    parser = argparse.ArgumentParser(description='Pyramidbox face Detector Training With MindSpore')
    parser.add_argument('--basenet', default='vgg16.ckpt', help='Pretrained base model')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--device_target', dest='device_target', help='device for training',
                        choices=['GPU', 'Ascend'], default='GPU', type=str)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--distribute', default=False, type=bool, help='Use mutil Gpu training')
    parser.add_argument('--save_folder', default='checkpoints/', help='Directory for saving checkpoint models')
    parser.add_argument('--epoches', default=100, type=int, help="Epoches to train model")
    parser.add_argument('--val_mindrecord', default='data/val.mindrecord', type=str, help="Path of val mindrecord file")
    args_ = parser.parse_args()
    return args_

def train(args):
    print("The argument is: ", args)
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE)
    device_id = 0
    device_num = 1
    ckpt_folder = os.path.join(args.save_folder, 'distribute_0')
    if args.distribute:
        D.init()
        device_id = get_rank()
        device_num = get_group_size()
        if device_id == 0 and not os.path.exists(ckpt_folder):
            os.mkdir(ckpt_folder)

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)

    else:
        context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))

    # Create train dataset
    ds_train = create_train_dataset(cfg, args.batch_size, device_num, device_id, args.num_workers)

    # Create val dataset
    ds_val = create_val_dataset(args.val_mindrecord, args.batch_size, 1, 0, args.num_workers)

    steps_per_epoch = ds_train.get_dataset_size()
    net = build_net("train", cfg.NUM_CLASSES)

    # load pretrained vgg16
    vgg_params = load_checkpoint(args.basenet)
    load_param_into_net(net.vgg, vgg_params)

    network = NetWithLoss(net)
    network.set_train(True)

    if args.distribute:
        milestone = cfg.DIS_LR_STEPS + [args.epoches * steps_per_epoch]
    else:
        milestone = cfg.LR_STEPS + [args.epoches * steps_per_epoch]

    learning_rates = [args.lr, args.lr * 0.1, args.lr * 0.01, args.lr * 0.001]
    lr_scheduler = nn.piecewise_constant_lr(milestone, learning_rates)

    optimizer = nn.SGD(params=network.trainable_params(), learning_rate=lr_scheduler, momentum=args.momentum,
                       weight_decay=args.weight_decay)

    # train net
    train_net = nn.TrainOneStepCell(network, optimizer)
    train_net.set_train(True)
    eval_net = EvalLoss(net)

    print("Start training net")
    whole_step = 0
    for epoch in range(1, args.epoches+1):
        step = 0
        time_list = []
        for d in ds_train.create_tuple_iterator():
            start_time = time.time()
            loss = train_net(*d)
            step += 1
            whole_step += 1
            print(f'epoch: {epoch} total step: {whole_step}, step: {step}, loss is {loss}')
            per_time = time.time() - start_time
            time_list.append(per_time)

        net.set_train(False)
        if args.distribute and device_id == 0:
            print('per step time: ', '%.2f' % (sum(time_list) / len(time_list) * 1000), "(ms/step)")
            val(epoch, eval_net, train_net, ds_val, ckpt_folder)

        elif not args.distribute:
            print('per step time: ', '%.2f' % (sum(time_list) / len(time_list) * 1000), "(ms/step)")
            val(epoch, eval_net, train_net, ds_val, args.save_folder)
        net.set_train(True)

def val(epoch, eval_net, model, ds_val, ckpt_dir):
    face_loss_list = []
    global MIN_LOSS
    for (images, face_loc, face_conf, _, _) in ds_val.create_tuple_iterator():
        face_loss = eval_net(images, face_loc, face_conf)
        face_loss_list.append(face_loss)

    a_loss = sum(face_loss_list) / len(face_loss_list)
    if a_loss < MIN_LOSS:
        MIN_LOSS = a_loss
        print("Saving best ckpt, epoch is ", epoch)
        save_checkpoint(model, os.path.join(ckpt_dir, f'pyramidbox_best_{epoch}.ckpt'))


if __name__ == '__main__':
    train_args = parse_args()
    set_seed(66)
    if not os.path.exists(train_args.save_folder):
        os.mkdir(train_args.save_folder)
    train(train_args)
