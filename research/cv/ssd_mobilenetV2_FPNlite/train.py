# pylint: disable=no-member
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

"""Train SSD and get checkpoint files."""

import os
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.train.summary.summary_record import SummaryRecord
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from src.ssd import SSDWithLossCell, TrainingWrapper, ssd_mobilenet_v2_fpn
from src.dataset import create_ssd_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter
from src.callbacks import CustomLossMonitor
from src.model_utils.config import config as cfg

set_seed(1)

def main():
    if cfg.modelarts_mode:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=cfg.run_platform,
                            device_id=device_id)
        cfg.coco_root = os.path.join(cfg.coco_root, str(device_id))
        cfg.mindrecord_dir = os.path.join(cfg.mindrecord_dir, str(device_id))
        if cfg.distribute:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              device_num=device_num)
            init()
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
            rank = get_rank()
        else:
            rank = 0
        if cfg.mindrecord_mode == "mindrecord":
            mox.file.copy_parallel(cfg.data_url, cfg.mindrecord_dir)
        else:
            mox.file.copy_parallel(cfg.data_url, cfg.coco_root)

    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg.run_platform)
        if cfg.distribute:
            if cfg.run_platform == "Ascend":
                if os.getenv("DEVICE_ID", "not_set").isdigit():
                    context.set_context(device_id=int(os.getenv("DEVICE_ID")))
                device_num = cfg.device_num
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True,
                                                  device_num=device_num)
                init()
                context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
                rank = get_rank()
            else:
                init()
                context.set_auto_parallel_context(
                    device_num=cfg.device_num,
                    parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True)
                rank = get_rank()
        else:
            rank = 0
            device_num = 1
            context.set_context(device_id=cfg.device_id)
    print(f"DATASET ARG = {cfg.dataset}")
    mindrecord_file = create_mindrecord(cfg.dataset, "ssd.mindrecord", True)
    if cfg.only_create_dataset:
        if cfg.modelarts_mode:
            mox.file.copy_parallel(cfg.mindrecord_dir, cfg.device_num)
        return

    loss_scale = float(cfg.loss_scale)

    # When create MindDataset, using the first mindrecord file, such as ssd.mindrecord0.
    dataset = create_ssd_dataset(mindrecord_file, repeat_num=1, batch_size=cfg.batch_size,
                                 device_num=cfg.device_num, rank=rank)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")


    ssd = ssd_mobilenet_v2_fpn(config=cfg)
    net = SSDWithLossCell(ssd, cfg)
    net.to_float(mstype.float32)
    init_net_param(net)

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * cfg.save_checkpoint_epochs)
    if cfg.distribute:
        save_ckpt_path = './ckpt/device_' + str(rank) + '/'
        summary_dir = './summary/device_' + str(rank) + '/'
    else:
        save_ckpt_path = './ckpt/'
        summary_dir = './summary/'
    ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=save_ckpt_path, config=ckpt_config)

    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.pre_trained)
        if cfg.filter_weight:
            filter_checkpoint_parameter(param_dict)
        load_param_into_net(net, param_dict)

    if cfg.freeze_layer == "backbone":
        for param in ssd.feature_extractor.mobilenet_v2.feature_1.trainable_params():
            param.requires_grad = False

    lr = Tensor(get_lr(global_step=cfg.pre_trained_epoch_size * dataset_size,
                       lr_init=cfg.lr_init, lr_end=cfg.lr_end_rate * cfg.lr, lr_max=cfg.lr,
                       warmup_epochs=cfg.warmup_epochs,
                       total_epochs=cfg.epoch_size,
                       steps_per_epoch=dataset_size))

    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay, loss_scale=1.0)

    net = TrainingWrapper(net, opt, loss_scale, True)


    with SummaryRecord(summary_dir) as summary_record:
        loss_cb = CustomLossMonitor(summary_record=summary_record, mode="train", frequency=5)
        callback = [TimeMonitor(data_size=dataset_size), loss_cb, ckpoint_cb]
        model = Model(net)
        dataset_sink_mode = False
        print("Start train SSD, the first epoch will be slower because of the graph compilation.")
        model.train(cfg.epoch_size, \
              dataset, \
              callbacks=callback, \
              dataset_sink_mode=dataset_sink_mode)
        if cfg.modelarts_mode:
            mox.file.copy_parallel(save_ckpt_path, cfg.train_url)

if __name__ == '__main__':
    main()
