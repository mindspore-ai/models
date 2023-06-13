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
from mindspore import nn
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore import Model, DynamicLossScaleManager, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.network import AutoEncoder, SSIMLoss, NetWithLoss
from src.dataset import Dataloader
from model_utils.config import config as cfg
from model_utils.device_adapter import get_device_id, get_device_num

set_seed(1234)


def train():
    dataloader = Dataloader()
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=get_device_id())
    rank = 0
    device_num = get_device_num()
    if cfg.distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num
        )
        init()
        rank = get_rank()
        cfg.batch_size = cfg.batch_size // device_num

    if rank == 0:
        if os.path.exists(cfg.aug_dir):
            remove_dir(cfg.aug_dir)
        os.makedirs(cfg.aug_dir)
        if os.path.exists(cfg.tmp):
            remove_dir(cfg.tmp)
        os.makedirs(cfg.tmp)
    train_dataset = dataloader.create_dataset(device_num, rank)
    cfg.dataset_size = train_dataset.get_dataset_size()
    loss = SSIMLoss()
    auto_encoder = AutoEncoder(cfg)
    net_loss = NetWithLoss(auto_encoder, loss)
    if cfg.load_ckpt_path != "":
        param_dict = load_checkpoint(cfg.load_ckpt_path)
        load_param_into_net(net_loss, param_dict)
        print("Load checkpoint {} to net successfully".format(cfg.load_ckpt_path))

    cb = [LossMonitor(1), TimeMonitor()]
    optimizer = nn.AdamWeightDecay(params=auto_encoder.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.decay)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.dataset_size, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="ssim_autocoder_" + cfg.dataset, directory="./checkpoint", config=ckpt_config)
    loss_scale_manager = DynamicLossScaleManager()

    if cfg.run_eval and rank == 0:
        from src.utils import get_results
        from src.eval_utils import EvalCallBack, apply_eval

        eval_cb = EvalCallBack(cfg, auto_encoder, get_results, apply_eval)
        cb += [eval_cb]

    model = Model(net_loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level="O0")
    if rank == 0:
        cb.append(ckpoint_cb)
    model.train(cfg.epochs, train_dataset, callbacks=cb, dataset_sink_mode=True)

    if cfg.model_arts and rank == 0:
        import moxing as mox

        mox.file.copy_parallel(src_url="./checkpoint", dst_url=cfg.train_url)


def remove_dir(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, True)


if __name__ == "__main__":
    train()
