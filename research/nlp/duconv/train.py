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
train model
"""
import os

from mindspore import context
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore import set_seed

from src.dataset import create_dataset
from src.model_utils.config import config as cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.model import RetrievalWithLoss
from src.bert import BertConfig
from src.lr_schedule import Noam

set_seed(0)


@moxing_wrapper()
def run_duconv():
    """run duconv task"""
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())

    config = BertConfig(seq_length=cfg.max_seq_length, vocab_size=cfg.vocab_size)
    epoch = cfg.epoch
    if cfg.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=cfg.device_id)
        if cfg.run_distribute:
            device_num = get_device_num()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        raise Exception("Target error, Ascend is supported.")

    use_kn = bool("kn" in cfg.task_name)

    if cfg.run_distribute:
        device_num = get_device_num()
        dataset = create_dataset(cfg.batch_size, data_file_path=cfg.train_data_file_path,
                                 device_num=device_num, rank=cfg.device_id,
                                 do_shuffle=(cfg.train_data_shuffle), use_knowledge=use_kn)
    else:
        dataset = create_dataset(cfg.batch_size, data_file_path=cfg.train_data_file_path,
                                 do_shuffle=(cfg.train_data_shuffle), use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()

    max_train_steps = cfg.epoch * steps_per_epoch
    warmup_steps = int(max_train_steps * cfg.warmup_proportion)
    network = RetrievalWithLoss(config, use_kn)
    lr_schedule = Noam(config.hidden_size, warmup_steps, cfg.learning_rate)
    optimizer = Adam(network.trainable_params(), lr_schedule)
    # network_one_step = TrainOneStepCell(network, optimizer)
    model = Model(network=network, optimizer=optimizer, amp_level="O2")

    data_size = cfg.save_checkpoint_steps if cfg.dataset_sink_mode else 100
    time_cb = TimeMonitor(data_size)
    loss_cb = LossMonitor(data_size)
    callbacks = [time_cb, loss_cb]
    # define callbacks
    if get_rank_id() == 0:
        if cfg.rank_save_ckpt_flag:
            ckpt_cfg = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,\
                keep_checkpoint_max=50)
        save_ckpt_path = os.path.join(cfg.save_checkpoint_path, 'ckpt_' + str(cfg.device_id) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_cfg, directory=save_ckpt_path,
                                  prefix=cfg.task_name + '_rank_' + str(get_device_id()))
        callbacks.append(ckpt_cb)
    if cfg.dataset_sink_mode:
        epoch = int(epoch * steps_per_epoch / data_size)
    model.train(epoch, dataset, callbacks, dataset_sink_mode=cfg.dataset_sink_mode, sink_size=data_size)


if __name__ == "__main__":
    run_duconv()
