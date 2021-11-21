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
"""Train HAKE"""

import os
import time
import numpy as np

from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import Adam
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint
from mindspore import context, Model, ParameterTuple, set_seed
import mindspore.dataset as ds

from src.HAKE_model import HAKE_GRAPH
from src.HAKE_for_train import HAKENetworkWithLoss_HEAD, HAKETrainOneStepCell, \
    HAKENetworkWithLoss_TAIL

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_num, get_rank_id
from src.dataset import DataReader, TrainDataset, BatchType

np.random.seed(config.random_seed)
set_seed(config.random_seed)


class HAKETimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.
        rank_id (int): current gpu id.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, data_size=None, rank_id=0):
        super(HAKETimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()
        self.rank_id = rank_id

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Monitor the time in training."""
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % 50 == 0:
            epoch_seconds = (time.time() - self.epoch_time) * 1000
            step_size = self.data_size
            cb_params = run_context.original_args()
            if hasattr(cb_params, "batch_num"):
                batch_num = cb_params.batch_num
                if isinstance(batch_num, int) and batch_num > 0:
                    step_size = cb_params.batch_num

            if not isinstance(step_size, int) or step_size < 1:
                raise ValueError("data_size must be positive int.")

            step_seconds = epoch_seconds / step_size

            print("epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds), flush=True)
            loss_file = config.save_path + "loss_{}.log"
            with open(loss_file.format(self.rank_id), "a+") as f:
                f.write("epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds))
                f.write('\n')


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
        rank_id (int): Current gpu id. Default: 0
        neg_type: (str): Current task negative type. head or tail. Default: head
    """

    def __init__(self, per_print_times=1, rank_id=0, neg_type="head"):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        self.type = neg_type
        self.epoch = 0

    def step_end(self, run_context):
        """Monitor the loss in training."""
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % 100 == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                "time:{}, context epoch:{}, step:{}, loss:{}, negative type:{}, gpu id:{}, current epoch:{}".format(
                    current_time,
                    cb_params.cur_epoch_num,
                    cb_params.cur_step_num,
                    str(cb_params.net_outputs),
                    self.type,
                    self.rank_id,
                    self.epoch))

            loss_file = config.save_path + "loss_{}.log"

            with open(loss_file.format(self.rank_id), "a+") as f:
                f.write("time: {}, context epoch: {}, step: {}, loss: {}, negative type: {}, current epoch: {}".format(
                    current_time,
                    cb_params.cur_epoch_num,
                    cb_params.cur_step_num,
                    str(cb_params.net_outputs),
                    self.type,
                    self.epoch))
                f.write('\n')


def run_hake_train():
    """run hake train """
    print(config)
    if config.do_distribute:
        init()
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        context.reset_auto_parallel_context()
        device_num = get_device_num()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        rank_id = get_rank_id()
        config.save_path = os.path.join(config.save_path, "ckpt_" + str(rank_id))
    else:
        device_num = 1
        rank_id = 0
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    if config.save_path and not os.path.exists(config.save_path):
        try:
            os.makedirs(config.save_path)
        except OSError:
            pass

    data_reader = DataReader(config.data_path)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    # base model
    hake = HAKE_GRAPH(num_entity, num_relation, config.hidden_dim, config.gamma, config.modulus_weight,
                      config.phase_weight)

    # training date
    if config.do_train:
        train_data_head = ds.GeneratorDataset(
            source=TrainDataset(data_reader, config.negative_sample_size, BatchType.HEAD_BATCH),
            column_names=["pos_triple", "neg_triples", "subsampling_weight"],
            shuffle=True, num_shards=device_num, shard_id=rank_id,
            num_parallel_workers=config.num_workers, max_rowsize=config.max_rowsize).batch(batch_size=config.batch_size,
                                                                                           drop_remainder=True)

        train_data_tail = ds.GeneratorDataset(
            source=TrainDataset(data_reader, config.negative_sample_size, BatchType.TAIL_BATCH),
            column_names=["pos_triple", "neg_triples", "subsampling_weight"],
            shuffle=True, num_shards=device_num, shard_id=rank_id,
            num_parallel_workers=config.num_workers, max_rowsize=config.max_rowsize).batch(batch_size=config.batch_size,
                                                                                           drop_remainder=True)

        # loss model
        netwithloss_head = HAKENetworkWithLoss_HEAD(hake, config.adversarial_temperature)
        netwithloss_tail = HAKENetworkWithLoss_TAIL(hake, config.adversarial_temperature)

        # Set training configuration
        current_learning_rate = config.learning_rate
        optimizer = Adam(ParameterTuple(netwithloss_tail.trainable_params()), current_learning_rate)

        head_callbacks = [HAKETimeMonitor(data_size=train_data_head.get_dataset_size(), rank_id=rank_id),
                          LossCallBack(rank_id=rank_id, neg_type="head")]
        tail_callbacks = [HAKETimeMonitor(data_size=train_data_tail.get_dataset_size(), rank_id=rank_id),
                          LossCallBack(rank_id=rank_id, neg_type="tail")]

        netwithgrads_head = HAKETrainOneStepCell(netwithloss_head, optimizer)
        netwithgrads_tail = HAKETrainOneStepCell(netwithloss_tail, optimizer)

        model_head = Model(netwithgrads_head)
        model_tail = Model(netwithgrads_tail)

        # save checkpoint and training
        ckpt_config_tail = CheckpointConfig(
            save_checkpoint_steps=train_data_tail.get_dataset_size() * config.save_skpt_epoch_every,
            keep_checkpoint_max=config.save_checkpoint_num,
            saved_network=hake)
        ckpoint_cb_tail = ModelCheckpoint(directory=config.save_path, config=ckpt_config_tail)
        tail_callbacks.append(ckpoint_cb_tail)

        model_head.train(config.max_epochs, train_data_head, head_callbacks, dataset_sink_mode=False)
        model_tail.train(config.max_epochs, train_data_tail, tail_callbacks, dataset_sink_mode=False)


if __name__ == '__main__':
    run_hake_train()
