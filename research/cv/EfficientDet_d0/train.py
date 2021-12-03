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
""" train efficientDet_b0"""
import os
import time
import argparse
import mindspore.nn as nn
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net
from src.config import config
from src.dataset import create_EfficientDet_datasets
from src.lr_schedule import get_lr_cosine
from src.utils import init_weights
from src.backbone import EfficientDetBackbone
from src.efficientdet.loss import FocalLoss
from src.dataset import create_mindrecord
set_seed(1)

class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss):
        super(WithLossCell, self).__init__()
        self.backbone = backbone
        self.loss = loss

    def construct(self, x, y):
        _, reg, cls, anchor = self.backbone(x)
        cls_loss, reg_loss = self.loss(reg, cls, anchor, y)
        return cls_loss + reg_loss

def main():
    parser = argparse.ArgumentParser(description="EfficientDet training")
    parser.add_argument("--distribute", type=bool, default=True, help="Run distribute, default is False.")
    parser.add_argument("--data_url", type=str, default=None, help="mindrecord dir")
    parser.add_argument("--train_url", type=str, default=None, help="ckpt output dir in obs")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices="Ascend",
                        help="run platform, only support Ascend.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="checkpoint path.")
    parser.add_argument('--is_modelarts', type=str, default="False", help='is train on modelarts')

    args_opt = parser.parse_args()

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv("RANK_SIZE"))

    loss_scale = config.loss_scale
    init_lr = config.lr
    args_opt.distribute = device_num > 1

    if args_opt.run_platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        if args_opt.distribute:
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
    else:
        raise ValueError("Unsupported platform.")

    checkpoint_path = args_opt.checkpoint_path
    if args_opt.is_modelarts == "True":
        import moxing as mox
        local_data_url = "/cache/data/" + str(device_id)
        mox.file.make_dirs(local_data_url)
        local_train_url = "/cache/ckpt"
        mox.file.make_dirs(local_train_url)
        filename = "EfficientDet.mindrecord0"
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        local_data_path = os.path.join(local_data_url, filename)
        rank_id = device_id

        if args_opt.checkpoint_path:
            checkpoint_path = "/cache/resume/" + str(device_id)
            mox.file.make_dirs(checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_path, "efdet.ckpt")
            mox.file.copy(args_opt.checkpoint_path, checkpoint_path)
    else:
        rank_id = int(os.getenv('RANK_ID'), 0)
        if not os.path.exists(config.mindrecord_dir):
            sync_lock = "/tmp/sync_create_mindrecord.lock"
            if rank_id == 0 and not os.path.exists(sync_lock):
                create_mindrecord("coco", "EfficientDet.mindrecord", True)
                print("create mindrecord file done.")
                try:
                    os.mknod(sync_lock)
                except IOError:
                    pass
            while True:
                if os.path.exists(sync_lock):
                    break
                time.sleep(10)

        local_data_path = os.path.join(config.mindrecord_dir, "EfficientDet.mindrecord0")

        local_train_url = config.save_checkpoint_path

    dataset = create_EfficientDet_datasets(local_data_path, repeat_num=1,
                                           num_parallel_workers=config.workers,
                                           batch_size=config.batch_size, device_num=device_num, rank=rank_id)
    dataset_size = dataset.get_dataset_size()

    print("Create dataset done!")

    net = EfficientDetBackbone(config.num_classes, 0, False, True)
    net.set_train()
    net.to_float(ms.float32)

    if checkpoint_path:
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(net, param_dict)
    else:
        init_weights(net)

    loss = FocalLoss()

    net_withloss = WithLossCell(net, loss)

    lr = Tensor(get_lr_cosine(init_lr=init_lr, steps_per_epoch=dataset_size, warmup_epochs=50,
                              max_epoch=config.epoch_size, t_max=config.epoch_size, eta_min=0.0))

    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                      config.momentum, config.weight_decay, loss_scale=loss_scale)

    net_with_grads = nn.TrainOneStepCell(net_withloss, optimizer=opt, sens=loss_scale)

    model = Model(net_with_grads, amp_level="O0")

    cb = [LossMonitor(), TimeMonitor()]

    config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)

    ckpt_cb = ModelCheckpoint(prefix="EfficientDet", directory=local_train_url, config=config_ck)
    print("============== Starting Training ==============")

    if device_id == 0:
        cb += [ckpt_cb]
        if args_opt.is_modelarts == "True":

            class TransferCallback(Callback):
                """ transfer callback used for modelarts """

                def __init__(self, local_train_path, obs_train_path):
                    super(TransferCallback, self).__init__()
                    self.local_train_path = local_train_path
                    self.obs_train_path = obs_train_path

                def step_end(self, run_context):
                    cb_params = run_context.original_args()
                    current_epoch = cb_params.cur_epoch_num
                    if current_epoch % 10 == 0 and current_epoch != 0:
                        mox.file.copy_parallel(self.local_train_path, self.obs_train_path)

            transferCb = TransferCallback(local_train_url, args_opt.train_url)
            cb += [transferCb]

    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    print("============== End Training ==============")


if __name__ == '__main__':

    main()
