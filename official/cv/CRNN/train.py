# Copyright 2020-21 Huawei Technologies Co., Ltd
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
"""crnn training"""
import os
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.loss import CTCLoss
from src.dataset import create_dataset
from src.crnn import crnn
from src.crnn_for_train import TrainOneStepCellWithGradClip
from src.metric import CRNNAccuracy
from src.eval_callback import EvalCallBack
from src.logger import get_logger
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id
from src.model_utils.callback import ResumeCallback, CRNNMonitor
from src.model_utils.lr_scheduler import cosine_decay_lr_with_start_step

set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)


def apply_eval(eval_param):
    evaluation_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = evaluation_model.eval(eval_ds)
    return res[metrics_name]


def modelarts_pre_process():
    pass


def set_default():
    config.rank_id = int(os.getenv('RANK_ID', '0'))
    config.log_dir = os.path.join(config.output_dir, 'log', 'rank_%s' % config.rank_id)
    config.save_ckpt_dir = os.path.join(config.output_dir, 'ckpt')


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    set_default()
    config.logger = get_logger(config.log_dir, config.rank_id)
    config.logger.info("config : %s", config)
    config.logger.info("Please check the above information for the configurations")

    if config.device_target == 'Ascend':
        device_id = get_device_id()
        context.set_context(device_id=device_id)

    if config.model_version == 'V1' and config.device_target != 'Ascend':
        raise ValueError("model version V1 is only supported on Ascend, pls check the config.")

    # lr_scale = 1
    if config.run_distribute:
        if config.device_target == 'Ascend':
            init()
            # lr_scale = 1
            device_num = get_device_num()
            rank = get_rank_id()
        else:
            init()
            # lr_scale = 1
            device_num = get_group_size()
            rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        device_num = 1
        rank = 0

    if config.resume_ckpt:
        resume_param = load_checkpoint(config.resume_ckpt,
                                       choice_func=lambda x: not x.startswith(('learning_rate', 'global_step')))
        config.train_start_epoch = int(resume_param.get('epoch_num', 0).asnumpy().item())
        config.logger.info("train_start_epoch: %d", config.train_start_epoch)
    max_text_length = config.max_text_length
    # create dataset
    dataset = create_dataset(name=config.train_dataset, dataset_path=config.train_dataset_path,
                             batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank, config=config)
    config.steps_per_epoch = dataset.get_dataset_size()
    config.logger.info("per_epoch_step_size: %d", config.steps_per_epoch)
    # define lr
    lr_init = config.learning_rate
    lr = cosine_decay_lr_with_start_step(0.0, lr_init, config.epoch_size * config.steps_per_epoch,
                                         config.steps_per_epoch, config.epoch_size,
                                         config.train_start_epoch * config.steps_per_epoch)
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    net = crnn(config, full_precision=config.device_target != 'Ascend')
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum, nesterov=config.nesterov)
    net_with_loss = WithLossCell(net, loss)
    net_with_grads = TrainOneStepCellWithGradClip(net_with_loss, opt).set_train()
    if config.resume_ckpt:
        load_param_into_net(net_with_grads, resume_param)
    # define model
    model = Model(net_with_grads)
    # define callbacks
    callbacks = [ResumeCallback(start_epoch=config.train_start_epoch)]
    if config.run_eval and rank == 0:
        if config.train_eval_dataset_path is None or (not os.path.isdir(config.train_eval_dataset_path)):
            raise ValueError("{} is not a existing path.".format(config.train_eval_dataset_path))
        eval_dataset = create_dataset(name=config.train_eval_dataset,
                                      dataset_path=config.train_eval_dataset_path,
                                      batch_size=config.batch_size,
                                      is_training=False,
                                      config=config)
        eval_model = Model(net, loss, metrics={'CRNNAccuracy': CRNNAccuracy(config, print_flag=False)})
        eval_param_dict = {"model": eval_model, "dataset": eval_dataset, "metrics_name": "CRNNAccuracy"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=config.save_ckpt_dir, best_ckpt_name="best_acc.ckpt",
                               eval_all_saved_ckpts=config.eval_all_saved_ckpts, metrics_name="acc")
        callbacks += [eval_cb]
    if config.save_checkpoint and rank == 0:
        ckpt_append_info = [{'epoch_num': 0}]
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max, append_info=ckpt_append_info)
        ckpt_cb = ModelCheckpoint(prefix="crnn", directory=config.save_ckpt_dir, config=config_ck)
        callbacks.append(ckpt_cb)
    callbacks.append(CRNNMonitor(config, lr))
    model.train(config.epoch_size - config.train_start_epoch, dataset, callbacks=callbacks,
                dataset_sink_mode=config.device_target == 'Ascend')


if __name__ == '__main__':
    train()
