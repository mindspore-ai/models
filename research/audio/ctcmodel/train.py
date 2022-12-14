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

"""ctc training"""

import os
from mindspore import context, Model, nn
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import CTCModel
from src.dataset import create_dataset
from src.loss import CTC_Loss
from src.model_for_train import WithCtcLossCell, TrainingWrapper
from src.model_for_eval import CTCEvalModel
from src.metric import LER
from src.eval_callback import EvalCallBack
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id


def modelarts_pre_process():
    config.train_path = config.local_train_path
    config.test_path = config.local_test_path
    config.save_dir = config.local_train_url
    if config.checkpoint_path:
        config.checkpoint_path = config.local_checkpoint_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """train_function"""

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == 'Ascend':
        if not config.enable_modelarts and not config.run_distribute:
            device_id = config.device_id
        else:
            device_id = get_device_id()
        context.set_context(device_id=device_id)

        if config.run_distribute:
            device_num = get_device_num()
            device_id = get_rank_id()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()

    elif config.device_target == 'GPU':
        if config.run_distribute:
            init()
            device_id = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = device_id
        else:
            device_num = 1
            device_id = config.device_id
            context.set_context(device_id=device_id)
            rank = 0

    ds_train = create_dataset(config.train_path, True, config.batch_size, num_shards=device_num, shard_id=rank)
    net = CTCModel(input_size=config.feature_dim, batch_size=config.batch_size, hidden_size=config.hidden_size,
                   num_class=config.n_class, num_layers=config.n_layer)
    if config.checkpoint_path:
        ckpt_file = config.checkpoint_path
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(net, param_dict)
    loss_fn = CTC_Loss(batch_size=config.batch_size, max_label_length=config.max_label_length)
    loss_net = WithCtcLossCell(net, loss_fn)
    step_size = ds_train.get_dataset_size()
    lr = nn.dynamic_lr.cosine_decay_lr(0.0, config.lr_init, config.epoch * step_size, step_size, config.epoch)
    opt = nn.Adam(net.trainable_params(), learning_rate=lr, eps=1e-3)
    train_net = TrainingWrapper(loss_net, opt, clip_global_norm_value=config.clip_value)
    train_net.set_train()
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if config.save_check and rank == 0:
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="ctc", directory=config.save_dir, config=config_ck)
        callbacks.append(ckpoint_cb)
    eval_net = CTCEvalModel(net)
    model = Model(train_net, eval_network=eval_net, metrics={'ler': LER(beam=config.beam)})
    if config.train_eval and rank == 0:
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        ds_test = create_dataset(config.test_path, False, config.batch_size)
        eval_callback = EvalCallBack(model=model, eval_ds=ds_test, interval=config.interval,
                                     ckpt_directory=config.save_dir)
        callbacks.append(eval_callback)
    model.train(config.epoch, ds_train, callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode)


if __name__ == "__main__":
    run_train()
