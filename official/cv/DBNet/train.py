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
"""DBNet Train."""
import os
import sys

import mindspore as ms
from mindspore import nn
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
import src.modules.loss as loss
from src.modules.model import get_dbnet, WithLossCell, TrainOneStepCell
from src.utils.callback import DBNetMonitor, ResumeCallback
from src.utils.learning_rate import warmup_polydecay
from src.utils.env import init_env
from src.utils.logger import get_logger
from src.datasets.load import create_dataset
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def init_group_params(net, weight_decay):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


def set_default():
    config.output_dir = os.path.join(config.output_dir, config.net, config.backbone.initializer)
    config.save_ckpt_dir = os.path.join(config.output_dir, 'ckpt')
    config.log_dir = os.path.join(config.output_dir, 'log')
    os.makedirs(config.save_ckpt_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)


@moxing_wrapper()
def train():
    set_default()
    init_env(config)
    if config.run_profiler:
        from mindspore.profiler import Profiler
        profiler = Profiler(output_path='./profiler_data')
    config.logger = get_logger(config.log_dir, config.rank_id)
    train_dataset, steps_pre_epoch = create_dataset(config, True)
    config.steps_per_epoch = steps_pre_epoch

    # Model
    net = get_dbnet(config.net, config, isTrain=True)
    if config.train.pretrained_ckpt:
        ms.load_checkpoint(net, config.train.pretrained_ckpt)
        config.logger.info("load pretrained checkpoint: %s", config.train.pretrained_ckpt)

    if config.train.resume_ckpt:
        resume_param = ms.load_checkpoint(config.train.resume_ckpt,
                                          choice_func=lambda x: not x.startswith(('learning_rate', 'global_step')))
        config.train.start_epoch_num = int(resume_param.get('epoch_num', ms.Tensor(0, ms.int32)).asnumpy().item())

    lr = ms.Tensor(warmup_polydecay(base_lr=config.optimizer.lr.base_lr,
                                    target_lr=config.optimizer.lr.target_lr,
                                    warmup_epoch=config.optimizer.lr.warmup_epoch,
                                    total_epoch=config.train.total_epochs,
                                    start_epoch=config.train.start_epoch_num,
                                    steps_pre_epoch=steps_pre_epoch,
                                    factor=config.optimizer.lr.factor))
    if config.optimizer.type == "momentum":
        config.logger.info("Use Momentum")
        opt = nn.Momentum(params=init_group_params(net, config.optimizer.weight_decay),
                          learning_rate=lr,
                          momentum=config.optimizer.momentum)
    elif config.optimizer.type == "adam":
        if hasattr(nn.Adam, "use_amsgrad"):
            config.logger.info("Use amsgrad Adam")
            opt = nn.Adam(net.trainable_params(), learning_rate=lr, use_amsgrad=True)
        else:
            config.logger.info("Use Adam")
            opt = nn.Adam(net.trainable_params(), learning_rate=lr)
    else:
        raise ValueError(f"Not support optimizer: {config.optimizer.type}")
    # Loss function
    criterion = loss.L1BalanceCELoss(eps=config.loss.eps, l1_scale=config.loss.l1_scale,
                                     bce_scale=config.loss.bce_scale, bce_replace=config.loss.bce_replace)
    if config.mix_precision:
        # only resnet run with float16
        net.to_float(ms.float32)
        net.backbone.to_float(ms.float16)
    net_with_loss = WithLossCell(net, criterion)
    train_net = TrainOneStepCell(net_with_loss, optimizer=opt, scale_sense=nn.FixedLossScaleUpdateCell(1024.),
                                 clip_grad=config.train.clip_grad, force_update=config.train.force_update)

    cb_default = list()
    # Recovery must be activated when not run evaluation
    enabel_recovery = config.enabel_recovery if config.run_eval else True
    if config.rank_id == 0 and enabel_recovery:
        ckpt_append_info = [{'epoch_num': 0}]
        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_pre_epoch * 50,
                                       keep_checkpoint_max=config.train.max_checkpoints, append_info=ckpt_append_info)
        cb_default.append(ModelCheckpoint(config=ckpt_config, directory=config.save_ckpt_dir,
                                          prefix=config.net + '-' + config.backbone.initializer))
    if config.train.resume_ckpt:
        ms.load_param_into_net(train_net, resume_param)
        cb_default.append(ResumeCallback(config.train.start_epoch_num))
        config.logger.info("Resume train from epoch: %s", config.train.start_epoch_num)
    cb_default.append(DBNetMonitor(config, net, lr.asnumpy(), per_print_times=config.per_print_times))
    model = ms.Model(train_net)
    config.logger.save_args(config)
    if config.run_profiler:
        model.train(3, train_dataset, callbacks=cb_default, sink_size=20,
                    dataset_sink_mode=config.train.dataset_sink_mode)
        profiler.analyse()
    else:
        model.train(config.train.total_epochs - config.train.start_epoch_num, train_dataset, callbacks=cb_default,
                    dataset_sink_mode=config.train.dataset_sink_mode)
    config.logger.info("Train has completed.")


if __name__ == '__main__':
    train()
