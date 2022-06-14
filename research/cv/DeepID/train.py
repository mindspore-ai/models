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
"""Train the model."""
from time import time
import os
import argparse
import ast

from mindspore import nn, save_checkpoint
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank

from src.dataset import dataloader
from src.utils import get_network, eval_func
from src.loss import DeepIDLoss
from src.cell import TrainOneStepCell
from src.report import Reporter

parser = argparse.ArgumentParser(description='DeepID')

parser.add_argument('--data_url', type=str, default='./data/', help='Dataset path')
parser.add_argument('--train_url', type=str, default='./data/', help='Train output path')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--input_dim', type=int, default=3, help='image dim')
parser.add_argument('--num_class', type=int, default=1283, help='number of classes')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument("--ckpt_path", type=str, default='./ckpt_path_11/', help="Checkpoint saving path.")
parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: 0.")
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--device_target", type=str, default='GPU', help="device target")
parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
parser.add_argument('--modelarts', type=ast.literal_eval, default=False, help='Dataset path')


if __name__ == '__main__':

    args_opt = parser.parse_args()
    if args_opt.modelarts:
        import moxing as mox

        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)
        context.set_context(device_id=device_id)
        local_data_url = './cache/data'
        local_train_url = '/cache/ckpt'

        local_data_url = os.path.join(local_data_url, str(device_id))
        local_train_url = os.path.join(local_train_url, str(device_id))

        # unzip data
        path = os.getcwd()
        print("cwd: %s" % path)
        data_url = 'obs://lxq/deepID/data/'

        data_name = '/crop_images_DB.zip'
        print('listdir1: %s' % os.listdir('./'))

        a1time = time()
        mox.file.copy_parallel(data_url, local_data_url)
        print('listdir2: %s' % os.listdir(local_data_url))
        b1time = time()
        print('time1:', b1time - a1time)

        a2time = time()
        zip_command = "unzip -o %s -d %s" % (local_data_url + data_name, local_data_url)
        if os.system(zip_command) == 0:
            print('Successful backup')
        else:
            print('FAILED backup')
        b2time = time()
        print('time2:', b2time - a2time)
        print('listdir3: %s' % os.listdir(local_data_url))
        dataset_path = local_data_url

    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                            device_id=args_opt.device_id, save_graphs=False)
        dataset_path = args_opt.data_url
        rank = 0
        if args_opt.run_distribute:
            device_num = args_opt.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()

            rank = get_rank()

    train_dataset, train_datalength = dataloader(dataset_path, args_opt.epochs, device_num=args_opt.device_num,
                                                 mode='train', batch_size=args_opt.batch_size, rank=args_opt.rank_id)

    val_dataset, val_datalength = dataloader(dataset_path, args_opt.epochs, device_num=args_opt.device_num,
                                             mode='valid', batch_size=args_opt.batch_size, rank=args_opt.rank_id)


    train_dataset_iter = train_dataset.create_dict_iterator()
    val_dataset_iter = val_dataset.create_dict_iterator()

    deepid = get_network(args_opt, args_opt.num_class)

    loss_cell = DeepIDLoss(deepid)

    opt = nn.Adam(deepid.trainable_params(), learning_rate=args_opt.lr)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    deepid_TrainOneStep = TrainOneStepCell(loss_cell, opt)

    deepid_TrainOneStep.set_train()

    if not os.path.exists(args_opt.ckpt_path):
        os.mkdir(args_opt.ckpt_path)

    epoch_iter = int(train_datalength / args_opt.batch_size)
    num_iters = int(epoch_iter * args_opt.epochs / args_opt.device_num)
    epoch_iter = int(epoch_iter / args_opt.device_num)
    start = time()
    reporter = Reporter(num_iters)
    best_acc = 0
    epoch = 0
    print('Start Training')
    for iterator in range(num_iters):
        current_time = time()
        data = next(train_dataset_iter)
        img = data['image']
        label = data['label']
        loss, acc = deepid_TrainOneStep(img, label)
        step_time = time()-current_time
        reporter.print_info(start, iterator, [loss, acc])
        print('Per step time: {:.2f} ms'.format(step_time*1000))
        if (iterator+1) % epoch_iter == 0:
            epoch += 1
            print('Start Validing')
            valid_acc = 0
            val_iter = 0
            for val in val_dataset_iter:
                val_iter += 1
                val_img = val['image']
                val_label = val['label']
                acc = eval_func(deepid, val_img, val_label)
                valid_acc += acc
                print('EVAL acc: ', acc*100, '%')
            aver_acc = valid_acc / val_iter
            print('Average Acc in Valid dataset: ', aver_acc*100, '%')
            if aver_acc >= best_acc:
                if rank == 0:
                    best_acc = aver_acc
                    save_checkpoint(deepid, args_opt.ckpt_path+"ckpt_deepid_best_{}.ckpt".format(rank))
    if args_opt.modelarts:
        mox.file.copy_parallel(save_checkpoint_path, args_opt.train_url)
