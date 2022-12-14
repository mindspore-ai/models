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

import os
import json
import time

from mindspore import nn, context
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset as ds
from mindspore.nn.optim import Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.utils.opts import parse_opts
from src.datasets.dataset import get_training_set, get_val_set
from src.models.non_local import I3DResNet50
from src.utils.callback import SaveCallback
from src.utils.lr_scheduler import MultiStepLR, get_lr
from src.models.resnet import resnet56, init_weight, init_group_prams

if __name__ == '__main__':
    # init options
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
    opt.arch = 'nl-{}'.format(opt.dataset)

    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    opt.result_path = os.path.join(opt.result_path, opt.arch, dir_time)
    print(opt)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # init context
    set_seed(opt.manual_seed)
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    if opt.modelarts:
        import moxing as mox

        device_id = int(os.getenv('DEVICE_ID'))
        local_data_path = '/cache/data'
        local_data_path = os.path.join(local_data_path, str(device_id))
        mox.file.copy_parallel(src_url=opt.data_url, dst_url=local_data_path)
        tar_command = "tar -xf " + os.path.join(local_data_path, "cifar-10.tar.gz") + " -C " + local_data_path
        opt.train_data_path = os.path.join(local_data_path, 'cifar-10-batches-bin')
        opt.test_data_path = os.path.join(local_data_path, 'cifar-10-batches-bin')
    if opt.distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        rank_id = get_rank()
        rank_size = get_group_size()
    else:
        context.set_context(device_id=opt.device_id)
        rank_id = 0
        rank_size = 1

    # define net
    assert opt.dataset in ['kinetics', 'cifar10']
    if opt.dataset == 'kinetics':
        net = I3DResNet50(pretrained_ckpt=opt.pretrained_ckpt)
    else:
        if opt.nl:
            print("ResNet-56 with non-local block after second residual block..")
            net = resnet56(non_local=True)
        else:
            print("ResNet-56 without non-local block..")
            net = resnet56(non_local=False)
        init_weight(net, 'XavierUniform', 'TruncatedNormal')

    if opt.dataset == 'kinetics':
        # define loss
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        # create dataset
        opt.video_path = opt.train_data_path
        training_data = get_training_set(opt)

        # define optimizer
        iters_per_epoch = training_data.get_dataset_size()
        scheduler = MultiStepLR(lr=opt.learning_rate, milestones=[40, 80], gamma=0.1, steps_per_epoch=iters_per_epoch,
                                max_epoch=120, warmup_epochs=0)
        lr = Tensor(scheduler.get_lr())
        optimizer = nn.SGD(
            params=net.trainable_params(),
            learning_rate=lr,
            momentum=opt.momentum,
            dampening=0 if opt.nesterov else opt.dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov
        )
        # create dataset
        opt.video_path = opt.test_data_path
        test_data = get_val_set(opt)
        # define model
        model = Model(network=net,
                      loss_fn=loss,
                      optimizer=optimizer,
                      metrics={'top_1_accuracy': nn.Top1CategoricalAccuracy(),
                               'top_5_accuracy': nn.Top5CategoricalAccuracy()}
                      )
    else:
        type_cast_op = C2.TypeCast(mstype.int32)
        transform_train = [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Rescale(1.0 / 255.0, 0.0),
            C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            C.HWC2CHW()
        ]
        if opt.distributed:
            training_data = ds.Cifar10Dataset(dataset_dir=opt.train_data_path, usage='train', shuffle=True,
                                              num_parallel_workers=opt.n_threads, num_shards=rank_size,
                                              shard_id=rank_id)
        else:
            training_data = ds.Cifar10Dataset(dataset_dir=opt.train_data_path, usage='train', shuffle=True,
                                              num_parallel_workers=opt.n_threads)
        training_data = training_data.map(transform_train, input_columns="image")
        training_data = training_data.map(type_cast_op, input_columns="label")
        training_data = training_data.batch(opt.batch_size // rank_size, drop_remainder=True)

        transform_test = [
            C.Rescale(1.0 / 255.0, 0.0),
            C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            C.HWC2CHW()
        ]
        # test dataset
        test_data = ds.Cifar10Dataset(dataset_dir=opt.test_data_path, usage='test', shuffle=False,
                                      num_parallel_workers=opt.n_threads)
        test_data = test_data.map(transform_test, input_columns="image")
        test_data = test_data.map(type_cast_op, input_columns="label")
        test_data = test_data.batch(opt.batch_size, drop_remainder=True)
        # define optimizer
        group_params = init_group_prams(net, opt.weight_decay)
        step_size = training_data.get_dataset_size()
        lr = Tensor(get_lr(lr_init=opt.learning_rate, lr_end=opt.learning_rate / 100, lr_max=opt.learning_rate * 10,
                           warmup_epochs=5, total_epochs=opt.n_epochs, steps_per_epoch=step_size,
                           lr_decay_mode="poly"))
        optimizer = Momentum(group_params, lr, opt.momentum, loss_scale=opt.loss_scale)

        # define loss
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss_scale = FixedLossScaleManager(opt.loss_scale, drop_overflow_update=False)

        # define model
        model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                      metrics={'top_1_accuracy': nn.Top1CategoricalAccuracy(),
                               'top_5_accuracy': nn.Top5CategoricalAccuracy()},
                      amp_level="O0", keep_batchnorm_fp32=False)

    # define callback
    step_size = training_data.get_dataset_size()
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size * opt.save_checkpoint_epochs,
                                 keep_checkpoint_max=10,
                                 saved_network=net)
    ckpt_cb = ModelCheckpoint(prefix=opt.arch, directory=opt.result_path, config=config_ck)
    cb = [time_cb, loss_cb]
    save_ckpt_cb = SaveCallback(model, test_data, opt)
    cb.append(save_ckpt_cb)

    # train
    model.train(opt.n_epochs, training_data, callbacks=cb, dataset_sink_mode=True)
    if opt.modelarts:
        mox.file.copy_parallel(src_url=opt.result_path, dst_url=opt.train_url)
