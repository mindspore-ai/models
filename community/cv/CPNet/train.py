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
""" train CPNet and get checkpoint files """
import os
from src.model import cpnet
from src.model.all_loss import All_Loss
from src.utils.lr import poly_lr
from src.utils.metric_and_evalcallback import cpnet_metric, EvalCallBack, CustomLossMonitor
from src.dataset.dataset import create_dataset
from src.model_utils.config import config as cfg
import mindspore
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.communication import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.summary import SummaryRecord

set_seed(1234)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cp_train():
    """ Train process """

    if cfg.device_target == "Ascend":
        bn = nn.SyncBatchNorm
    else:
        bn = nn.BatchNorm2d

    if cfg.distribute:
        if cfg.device_target == "Ascend":
            context.set_context(device_id=int(os.getenv("DEVICE_ID")))
        device_num = cfg.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          parameter_broadcast=True,
                                          device_num=device_num)
        init()
        rank = get_rank()
        CPNet = cpnet.CPNet(
            prior_channels=256,
            proir__size=60,
            am_kernel_size=11,
            pretrained=True,
            pretrained_path=cfg.pretrain_path,
            deep_base=True,
            BatchNorm_layer=bn
        )
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=int(0))
        CPNet = cpnet.CPNet(
            prior_channels=256,
            proir__size=60,
            am_kernel_size=11,
            pretrained=True,
            pretrained_path=cfg.pretrain_path,
            deep_base=True
        )

    if cfg.data_root[-1] == "/":
        train_list = cfg.data_root + cfg.train_list
        val_list = cfg.data_root + cfg.val_list
    else:
        train_list = cfg.data_root + '/' + cfg.train_list
        val_list = cfg.data_root + '/' + cfg.val_list
    train_dataset = create_dataset('train', cfg.data_root, train_list, device_num, rank)

    # validation

    validation_dataset = create_dataset('val', cfg.data_root, val_list, device_num, rank)

    # loss
    train_net_loss = All_Loss()

    steps_per_epoch = train_dataset.get_dataset_size()  # Return the number of batches in an epoch.
    total_train_steps = steps_per_epoch * cfg.epochs

    # get learning rate
    lr_iter = poly_lr(0.005, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    lr_iter_ten = poly_lr(0.05, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)

    pretrain_params = list(filter(lambda x: 'backbone' in x.name, CPNet.trainable_params()))
    cls_params = list(filter(lambda x: 'backbone' not in x.name, CPNet.trainable_params()))
    group_params = [{'params': pretrain_params, 'lr': Tensor(lr_iter, mindspore.float32)},
                    {'params': cls_params, 'lr': Tensor(lr_iter_ten, mindspore.float32)}]
    opt = nn.SGD(
        params=group_params,
        momentum=0.9,
        weight_decay=0.0001,
        loss_scale=1024,
    )
    # loss scale
    manager_loss_scale = FixedLossScaleManager(1024, False)

    m_metric = {'val_loss': cpnet_metric(cfg.classes, 255)}

    model = Model(
        CPNet, train_net_loss, optimizer=opt, loss_scale_manager=manager_loss_scale, metrics=m_metric
    )
    summary_save_dir = "./summary/"
    ckpt_save_dir = cfg.save_dir
    if cfg.distribute:
        ckpt_save_dir += "ckpt_" + str(rank) + '/'
        summary_save_dir = "./summary/device_" + str(rank) + '/'
    with SummaryRecord(summary_save_dir) as summary_record:

        # callback for saving ckpts
        time_cb = TimeMonitor(data_size=steps_per_epoch)
        loss_cb = CustomLossMonitor(summary_record=summary_record, mode="train", frequency=cfg.collection_freq)

        # validation
        epoch_per_eval = {"epoch": [], "val_loss": []}
        eval_cb = EvalCallBack(model, validation_dataset, 1, epoch_per_eval, summary_record, ckpt_dir=ckpt_save_dir)

        model.train(
            cfg.epochs, train_dataset, callbacks=[loss_cb, time_cb, eval_cb], dataset_sink_mode=False,
        )

    dict_eval = eval_cb.get_dict()
    val_num_list = dict_eval["epoch"]
    val_value = dict_eval["val_loss"]
    for i in range(len(val_num_list)):
        print(val_num_list[i], " : ", val_value[i])


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)

    cp_train()
