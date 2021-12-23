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
"""
    csd_train.py
"""
import os
import time
import datetime

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as numpy
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.common import set_seed
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore.communication.management import init

from src.args import args
from src.data.div2k import DIV2K
from src.edsr_slim import EDSR
from src.contras_loss import ContrastLoss

class NetWithLossCell(nn.Cell):
    """[NetWithLossCell]

    Args:
    """
    def __init__(self, net):
        super(NetWithLossCell, self).__init__()
        self.net = net
        self.l1_loss = nn.L1Loss()

    def construct(self, lr, hr, stu_width_mult, tea_width_mult):
        sr = self.net(lr, stu_width_mult)
        tea_sr = self.net(lr, tea_width_mult)
        loss = self.l1_loss(sr, hr) + self.l1_loss(tea_sr, hr)
        return loss

class NetWithCSDLossCell(nn.Cell):
    """[NetWithCSDLossCell]

    Args:
    """
    def __init__(self, net, contrast_w=0, neg_num=0):
        super(NetWithCSDLossCell, self).__init__()
        self.net = net
        self.neg_num = neg_num
        self.l1_loss = nn.L1Loss()
        self.contrast_loss = ContrastLoss()
        self.contrast_w = contrast_w

    def construct(self, lr, hr, stu_width_mult, tea_width_mult):
        """construct"""
        sr = self.net(lr, stu_width_mult)
        tea_sr = self.net(lr, tea_width_mult)
        loss = self.l1_loss(sr, hr) + self.l1_loss(tea_sr, hr)

        if self.contrast_w > 0:
            resize = nn.ResizeBilinear()
            bic = resize(lr, size=(lr.shape[-2] * 4, lr.shape[-1] * 4))
            neg = numpy.flip(bic, 0)
            neg = neg[:self.neg_num, :, :, :]
            loss += self.contrast_w * self.contrast_loss(tea_sr, sr, neg)
        return loss

class TrainOneStepCell(nn.Cell):
    """[TrainOneStepCell]

    Args:
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, lr, hr, width_mult, tea_width_mult):
        """construct"""
        weights = self.weights
        loss = self.network(lr, hr, width_mult, tea_width_mult)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(lr, hr, width_mult, tea_width_mult, sens)
        self.optimizer(grads)
        return loss

def ensure_path(path, remove=True):
    """
    ensure_path
    :param path:
    :param remove:
    :return:
    """
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                       or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def csd_train(train_loader, net, opt, did):
    """[csd_train]

    Args:
        did: device id
        train_loader ([type]): [description]
        net ([type]): [description]
        opt ([type]): [description]
    """
    set_seed(1)
    print("[CSD] Start Training...")

    step_size = train_loader.get_dataset_size()
    lr = []
    for i in range(0, opt.epochs):
        cur_lr = opt.lr / (2 ** ((i + 1) // 200))
        lr.extend([cur_lr] * step_size)
    optim = nn.Adam(net.trainable_params(), learning_rate=lr, loss_scale=opt.loss_scale)

    net_with_loss = NetWithCSDLossCell(net, args.contra_lambda, args.neg_num)
    train_cell = TrainOneStepCell(net_with_loss, optim)
    net.set_train()

    for epoch in range(0, opt.epochs):
        starttime = datetime.datetime.now()
        epoch_loss = 0
        for _, batch in enumerate(train_loader.create_dict_iterator(), 1):
            lr = batch["LR"]
            hr = batch["HR"]

            loss = train_cell(lr, hr, Tensor(opt.stu_width_mult), Tensor(1.0))
            epoch_loss += loss

        endtime = datetime.datetime.now()
        cost = (endtime - starttime).seconds
        print(f"time of epoch{epoch}: {cost}")
        print(f"Epoch[{epoch}] loss: {epoch_loss.asnumpy()}")

        if (epoch) % 10 == 0:
            print('===> Saving model...')
            if did == 0:
                save_checkpoint(net, os.path.join(opt.ckpt_save_path, f'{opt.filename}.ckpt'))


if __name__ == '__main__':
    print(args)
    time_start = time.time()
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target,
                        save_graphs=False, device_id=device_id)

    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, global_rank=device_id,
                                          gradients_mean=True)


    if args.modelArts_mode:
        import moxing as mox

        local_data_url = '/cache/data'
        print(f"[csd_train.py] local_data_url:{local_data_url}")
        args.dir_data = local_data_url
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)

        local_output_url = os.path.join('./ckpt', args.filename)
        ensure_path(local_output_url)
        args.ckpt_save_path = local_output_url

    if not os.path.exists(args.ckpt_save_path):
        print(f"Creating {args.ckpt_save_path}")
        os.makedirs(args.ckpt_save_path)
    train_dataset = DIV2K(args, name=args.data_train, train=True, benchmark=False)
    train_dataset.set_scale(args.task_id)
    print(len(train_dataset))
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)

    net_m = EDSR(args)
    print("Init net weights successfully")

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")

    csd_train(train_de_dataset, net_m, args, device_id)
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))

    if args.modelArts_mode:
        import moxing as mox
        mox.file.copy_parallel(src_url=args.ckpt_save_path, dst_url=args.train_url)
