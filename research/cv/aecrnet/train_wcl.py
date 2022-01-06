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
"""train"""
import os
import time
import datetime
import math
from mindspore import context, Tensor
from mindspore.context import ParallelMode
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.communication.management import init
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
import mindspore.numpy as numpy
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from src.args import args
from src.contras_loss import ContrastLoss
from src.models.model import Dehaze
from haze_data import RESIDEDatasetGenerator

class NetWithCRLossCell(nn.Cell):
    """[NetWithCRLossCell]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, net, contrast_w=0, neg_num=0):
        super(NetWithCRLossCell, self).__init__()
        self.net = net
        self.neg_num = neg_num
        self.l1_loss = nn.L1Loss()
        self.contrast_loss = ContrastLoss()
        self.contrast_w = contrast_w

    def construct(self, hazy, clear):
        pred = self.net(hazy)

        neg = numpy.flip(hazy, 0)
        neg = neg[:self.neg_num, :, :, :]
        l1_loss = self.l1_loss(pred, clear)
        contras_loss = self.contrast_loss(clear, pred, neg)
        loss = l1_loss + self.contrast_w * contras_loss
        return loss

class TrainOneStepCell(nn.Cell):
    """[TrainOneStepCell]

    Args:
        nn ([type]): [description]
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

    def construct(self, hazy, clear):
        weights = self.weights
        loss = self.network(hazy, clear)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(hazy, clear, sens)
        self.optimizer(grads)
        return loss


def train():
    """train"""
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, global_rank=device_id,
                                          gradients_mean=True)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_id=device_id)
    if args.modelArts_mode:
        import moxing as mox
        context.set_context(device_target="Ascend")
        local_data_url = '/cache/data'
        print(f"[train_wcl.py] local_data_url:{local_data_url}")
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        args.dir_data = local_data_url
        local_output_url = os.path.join('/home/work/user-job-dir/aecrnet/ckpt')
        if not os.path.exists(local_data_url):
            print(f"Creating {local_output_url}")
            os.makedirs(local_output_url)
        args.ckpt_save_path = local_output_url
    else:
        context.set_context(device_target=args.device_target)
    if not os.path.exists(args.ckpt_save_path):
        os.makedirs(args.ckpt_save_path)

    train_dataset = RESIDEDatasetGenerator(args, train=True)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["hazy", "gt"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)

    net_m = Dehaze(3, 3, rgb_range=args.rgb_range)
    print("Init net weights successfully")

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")
    step_size = train_de_dataset.get_dataset_size()
    lr = []
    for i in range(0, args.epochs):
        cur_lr = 0.5 * (1 + math.cos(i * math.pi / args.epochs)) * args.lr
        lr.extend([cur_lr] * step_size)
    opt = nn.Adam(net_m.trainable_params(), learning_rate=Tensor(lr, mstype.float32), loss_scale=args.loss_scale)

    net_with_loss = NetWithCRLossCell(net_m, args.contra_lambda, args.neg_num)
    train_cell = TrainOneStepCell(net_with_loss, opt)
    net_m.set_train()

    for epoch in range(0, args.epochs):
        starttime = datetime.datetime.now()
        epoch_loss = 0
        for _, batch in enumerate(train_de_dataset.create_dict_iterator(), 1):
            hazy = batch["hazy"]
            clear = batch["gt"]

            loss = train_cell(hazy, clear)
            epoch_loss += loss

        endtime = datetime.datetime.now()
        cost = (endtime - starttime).seconds
        print(f"time of epoch{epoch}: {cost}")
        print(f"Epoch[{epoch}] loss: {epoch_loss.asnumpy()}")

        if (epoch) % 10 == 0:
            if device_id == 0:
                print('===> Saving checkpoint...')
                save_checkpoint(net_m, os.path.join(args.ckpt_save_path, f'{args.filename}.ckpt'))

    if args.modelArts_mode:
        import moxing as mox
        mox.file.copy_parallel(src_url=args.ckpt_save_path, dst_url=args.train_url)

if __name__ == "__main__":
    time_start = time.time()
    train()
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))
