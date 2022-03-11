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

import time
import os
import math

import mindspore
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.communication import get_rank, get_group_size
from mindspore import save_checkpoint
import mindspore.nn as nn
import mindspore.dataset as ds

from src.dataset import DataIterator, create_dataset
from src.dien import DIEN, Ctr_Loss, CustomWithLossCell, TrainOneStepCell
from src.config import parse_args

# hyper-parameter
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
args_opt = parse_args()

if args_opt.is_modelarts:
    import moxing as mox
epoch_size = args_opt.epoch_size
base_lr = args_opt.base_lr
batch_size = args_opt.batch_size
max_len = args_opt.max_len


def train(ds_train, save_checkpoint_path,
          train_file,
          uid_voc,
          mid_voc,
          cat_voc,
          meta_path,
          review_path,
          batch_size_train=128,
          maxlen=100,
          ):
    # train data
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, meta_path, review_path,
                              batch_size_train, maxlen,
                              shuffle_each_epoch=False)

    n_uid, n_mid, n_cat = train_data.get_n()
    model = DIEN(n_uid, n_mid, n_cat, embedding_size=EMBEDDING_DIM)

    # loss function
    loss_fn = Ctr_Loss()
    net_with_criterion = CustomWithLossCell(model, loss_fn)

    dataset_size = ds_train.get_dataset_size()

    milestone = []
    learning_rates = []
    for i in range(epoch_size):
        milestone.append(dataset_size * (i + 1))
        learning_rates.append(base_lr * math.pow(0.5, i))
    lr = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)

    # optimizer
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    train_net = TrainOneStepCell(net_with_criterion, optimizer)

    print(ds_train.get_dataset_size())
    for epoch in range(epoch_size):
        time_start = time.time()
        pre_step_time = time_start
        step = 0
        print('epoch', epoch)
        loss_sum = Tensor(0.0, dtype=mstype.float32)
        for d in ds_train.create_dict_iterator():
            loss = train_net(d['mid_mask'], d['uids'], d['mids'], d['cats'], d['mid_his'], d['cat_his'],
                             d['noclk_mids'], d['noclk_cats'], d['target'])
            loss_sum += loss
            step_end = time.time()
            print("current_epoch:{0}, current_step:{1}, step_time:{2}".format(epoch, step, step_end - pre_step_time))
            pre_step_time = step_end
            step += 1
        time_end = time.time()
        print('train_loss:', loss_sum.asnumpy() / step)
        print('epoch_time:', time_end - time_start)
        if args_opt.dataset_type == 'Books':
            save_checkpoint(model, save_checkpoint_path + '/Books_DIEN{0}.ckpt'.format(epoch))
        elif args_opt.dataset_type == 'Electronics':
            save_checkpoint(model, save_checkpoint_path + '/Electronics_DIEN{0}.ckpt'.format(epoch))
        else:
            print('error:Dataset type must be Books or Electronics')


def modelarts():
    """modelarts"""
    if args_opt.run_distribute:
        device_num = int(os.getenv('RANK_SIZE'))
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, parameter_broadcast=True,
                                          gradients_mean=True)
        mindspore.communication.init()

        # define local data path
        dataset_type = args_opt.dataset_type
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_file_path = '/cache/dataset/device_' + os.getenv(
            'DEVICE_ID') + '/' + dataset_type + '/local_train_splitByUser'
        uid_voc_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/' + dataset_type + '/uid_voc.pkl'
        mid_voc_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/' + dataset_type + '/mid_voc.pkl'
        cat_voc_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/' + dataset_type + '/cat_voc.pkl'
        meta_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/' + dataset_type + '/item-info'
        review_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/' + dataset_type + '/reviews-info'
        if args_opt.dataset_type == 'Electronics':
            train_dataset_path = '/cache/dataset/device_' + os.getenv(
                'DEVICE_ID') + '/Electronics_train_1.mindrecord'
        else:
            train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/Books_train_1.mindrecord'
        ds_train = ds.MindDataset(train_dataset_path, num_parallel_workers=8,
                                  shuffle=True,
                                  num_shards=device_num,
                                  shard_id=device_id)
        save_checkpoint_path = '/cache/ckpt/device_' + os.getenv('DEVICE_ID')
        mox.file.make_dirs(save_checkpoint_path)
        mox.file.copy_parallel(src_url=args_opt.pretrained_ckpt_path, dst_url=save_checkpoint_path)
        train(ds_train, save_checkpoint_path, train_file=train_file_path, uid_voc=uid_voc_path,
              mid_voc=mid_voc_path, cat_voc=cat_voc_path, meta_path=meta_path, review_path=review_path,
              batch_size_train=batch_size, maxlen=max_len)

        print('Upload ckpt.')
        mox.file.copy_parallel(src_url=save_checkpoint_path, dst_url=args_opt.train_url)
    else:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)

        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/dataset_mindrecord')
        if args_opt.dataset_type == 'Electronics':
            train_dataset_path = '/cache/dataset/device_' + os.getenv(
                'DEVICE_ID') + '/Electronics_train_1.mindrecord'
        else:
            train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/Books_train_1.mindrecord'
        dataset_type = args_opt.dataset_type
        train_file_path = '/cache/dataset_mindrecord/' + dataset_type + '/local_train_splitByUser'
        uid_voc_path = '/cache/dataset_mindrecord/' + dataset_type + '/uid_voc.pkl'
        mid_voc_path = '/cache/dataset_mindrecord/' + dataset_type + '/mid_voc.pkl'
        cat_voc_path = '/cache/dataset_mindrecord/' + dataset_type + '/cat_voc.pkl'
        meta_path = '/cache/dataset_mindrecord/' + dataset_type + '/item-info'
        review_path = '/cache/dataset_mindrecord/' + dataset_type + '/reviews-info'
        ds_train = ds.MindDataset(train_dataset_path,
                                  num_parallel_workers=8, shuffle=True)
        save_checkpoint_path = '/cache/ckpt/'
        mox.file.make_dirs(save_checkpoint_path)
        mox.file.copy_parallel(src_url=args_opt.pretrained_ckpt_path, dst_url=save_checkpoint_path)
        train(ds_train, save_checkpoint_path, train_file=train_file_path, uid_voc=uid_voc_path,
              mid_voc=mid_voc_path, cat_voc=cat_voc_path, meta_path=meta_path, review_path=review_path,
              batch_size_train=batch_size, maxlen=max_len)
        print('Upload ckpt.')
        mox.file.copy_parallel(src_url=save_checkpoint_path, dst_url=args_opt.train_url)


def not_modelarts(target):
    """not_modelarts"""
    dataset_file_path = args_opt.dataset_file_path
    train_file = os.path.join(dataset_file_path, "local_train_splitByUser")
    uid_voc = os.path.join(dataset_file_path, "uid_voc.pkl")
    mid_voc = os.path.join(dataset_file_path, "mid_voc.pkl")
    cat_voc = os.path.join(dataset_file_path, "cat_voc.pkl")
    meta_path = os.path.join(dataset_file_path, "item-info")
    review_path = os.path.join(dataset_file_path, "reviews-info")
    save_checkpoint_path = './ckpt/'
    train_mindrecord_path = os.path.join(args_opt.mindrecord_path, '{0}_train.mindrecord'.format(args_opt.dataset_type))
    if args_opt.run_distribute:
        if target == 'Ascend':
            rank_id = get_rank()
            rank_size = get_group_size()
            ds_train = ds.MindDataset(train_mindrecord_path,
                                      num_shards=rank_size,
                                      shard_id=rank_id)
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              parameter_broadcast=True,
                                              gradients_mean=True,
                                              device_num=device_num)
            mindspore.communication.init()
            train(ds_train, save_checkpoint_path, train_file=train_file, uid_voc=uid_voc,
                  mid_voc=mid_voc, cat_voc=cat_voc, meta_path=meta_path, review_path=review_path,
                  batch_size_train=batch_size, maxlen=max_len)
        elif target == 'GPU':
            mindspore.communication.init("nccl")
            rank_size = get_group_size()
            rank_id = get_rank()
            context.set_context(device_id=rank_id)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              parameter_broadcast=True,
                                              gradients_mean=True)

            ds_train = ds.MindDataset(train_mindrecord_path,
                                      num_shards=rank_size,
                                      shard_id=rank_id)
            save_checkpoint_path += str(rank_id)
            if not os.path.exists(save_checkpoint_path):
                os.mkdir(save_checkpoint_path)
            train(ds_train, save_checkpoint_path, train_file=train_file, uid_voc=uid_voc,
                  mid_voc=mid_voc, cat_voc=cat_voc, meta_path=meta_path, review_path=review_path)
    else:
        if target == 'Ascend':
            device_id = args_opt.device_id
            context.set_context(device_id=device_id)
            ds_train = create_dataset(train_mindrecord_path)
            train(ds_train, save_checkpoint_path, train_file=train_file, uid_voc=uid_voc,
                  mid_voc=mid_voc, cat_voc=cat_voc, meta_path=meta_path, review_path=review_path,
                  batch_size_train=batch_size, maxlen=max_len)

        elif target == 'GPU':
            device_id = args_opt.device_id
            device_target = args_opt.device_target
            context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)
            ds_train = create_dataset(train_mindrecord_path)
            if not os.path.exists(save_checkpoint_path):
                os.mkdir(save_checkpoint_path)
            train(ds_train, save_checkpoint_path, train_file=train_file, uid_voc=uid_voc,
                  mid_voc=mid_voc, cat_voc=cat_voc, meta_path=meta_path,
                  review_path=review_path, batch_size_train=batch_size, maxlen=max_len)


def main():
    target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target)

    if args_opt.is_modelarts:
        # training in modelarts
        modelarts()
    else:
        not_modelarts(target)


if __name__ == '__main__':
    main()
