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
import shutil
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import nn
from mindspore import Model, DynamicLossScaleManager, context
from mindspore.ops import ReduceMean
from mindspore.context import ParallelMode
from mindspore.communication.management import init

from src.network import AutoEncoder
from src.dataset import Dataloader
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num
from model_utils.options import Options

class SSIMLoss(nn.Cell):
    def __init__(self, max_val=1.0):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val
        self.loss_fn = nn.SSIM(max_val=self.max_val)
        self.reduce_mean = ReduceMean()

    def construct(self, input_batch, target):
        output = self.loss_fn(input_batch, target)
        loss = 1 - self.reduce_mean(output)
        return loss


class NetWithLoss(nn.Cell):
    def __init__(self, net, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, input_batch):
        output = self._net(input_batch)
        return self._loss_fn(output, input_batch)


def train():

    options = Options()
    cfg = options.parse()
    dataloader = Dataloader()
    context.set_context(mode=context.GRAPH_MODE
                        , device_target=cfg["device_target"], device_id=get_device_id())
    rank = get_rank_id()
    device_num = get_device_num()
    if cfg["distribute"]:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL
                                          , gradients_mean=True,
                                          device_num=device_num
                                          , parameter_broadcast=True)
        init()
    train_dataset = dataloader.create_dataset(device_num, rank)
    loss = SSIMLoss()
    autoencoder = AutoEncoder(cfg)
    net_loss = NetWithLoss(autoencoder, loss)
    optimizer = nn.AdamWeightDecay(params=autoencoder.trainable_params(), learning_rate=cfg["lr"],
                                   weight_decay=cfg["decay"])
    ckpt_config = CheckpointConfig(save_checkpoint_steps=20,
                                   keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix='ssim_autocoder',
                                 directory=cfg["checkpoint_dir"],
                                 config=ckpt_config)
    loss_scale_manager = DynamicLossScaleManager()
    model = Model(net_loss, optimizer=optimizer
                  , loss_scale_manager=loss_scale_manager)
    if rank == 0:
        model.train(cfg["epochs"], train_dataset
                    , callbacks=[ckpoint_cb, LossMonitor(1), TimeMonitor()],
                    dataset_sink_mode=True)
    else:
        model.train(cfg["epochs"], train_dataset, callbacks=[LossMonitor(1), TimeMonitor()],
                    dataset_sink_mode=True)
    if cfg["model_arts"]:
        import moxing as mox
        mox.file.copy_parallel(src_url=cfg["checkpoint_dir"], dst_url=cfg["train_url"])
    if rank == 0:
        if os.path.exists(cfg["aug_dir"]):
            remove_dir(cfg["aug_dir"])
            if cfg["distribute"]:
                os.removedirs('../train_patches')
            else:
                os.removedirs('./train_patches')
        notice_copy_over = os.path.join(cfg["checkpoint_dir"], "copy_is_over")
        if os.path.exists(notice_copy_over):
            os.removedirs(notice_copy_over)

def remove_dir(path):
    file_list = os.listdir(path)
    for file in file_list:
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)

if __name__ == '__main__':
    train()
