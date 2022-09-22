# Copyright 2020 Huawei Technologies Co., Ltd
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
""" train model """
import os
import time
import numpy as np
import mindspore.nn as nn
from mindspore.nn.optim import AdamWeightDecay
from mindspore import Tensor, load_checkpoint, load_param_into_net
from mindspore import set_seed
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.common.dtype as mstype
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, Callback
from mindspore import save_checkpoint
from src.lr_schedule import create_dynamic_lr
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper, get_device_num
from src.model_utils.config import config as cfg, optimizer_cfg
from src.roberta_model import RobertaGenerationConfig, RobertaGenerationEncoder, RobertaGenerationDecoder
from src.model_encoder_decoder import EncoderDecoderModel
from src.model_train import EncoderDecoderWithLossCell, LabelSmoothedNllLoss
from src.dataset import create_dataset

set_seed(2022)


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))


time_stamp_init = False
time_stamp_first = 0

cfg.dtype = mstype.float32
cfg.compute_type = mstype.float16


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" %
                  (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)

        loss_file = "./loss_{}.log"
        if cfg.enable_modelarts:
            loss_file = "/cache/train/loss_{}.log"

        with open(loss_file.format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                loss,
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    cfg.train_data_dir = os.path.join(cfg.data_path, 'train.mindrecord')
    cfg.encoder_checkpoint_dir = os.path.join(
        cfg.load_path, 'robarta_base_encoder_ms.ckpt')
    cfg.decoder_checkpoint_dir = os.path.join(
        cfg.load_path, 'robarta_base_decoder_ms.ckpt')

    cfg.ckpt_save_dir = cfg.output_path
    print('modelarts_pre')


def get_config():
    cfg.train_data_dir = os.path.join(cfg.data_path, 'train.mindrecord')
    cfg.encoder_checkpoint_dir = os.path.join(
        cfg.checkpoint_path, 'robarta_base_encoder_ms.ckpt')
    cfg.decoder_checkpoint_dir = os.path.join(
        cfg.checkpoint_path, 'robarta_base_decoder_ms.ckpt')


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """ run train """
    if not cfg.enable_modelarts:
        get_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    cfg.device_num = get_device_num()
    if not os.path.exists(cfg.ckpt_save_dir):
        os.mkdir(cfg.ckpt_save_dir)
    if cfg.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if cfg.device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=cfg.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            save_ckpt_path = os.path.join(
                cfg.ckpt_save_dir, 'ckpt_' + str(get_rank()) + '/')
        else:
            save_ckpt_path = os.path.join(cfg.ckpt_save_dir, 'ckpt_0/')

    encoder_config = RobertaGenerationConfig(cfg)
    encoder = RobertaGenerationEncoder(encoder_config, add_pooling_layer=False)

    decoder_config = RobertaGenerationConfig(cfg, is_decoder=True, add_cross_attention=True)
    decoder = RobertaGenerationDecoder(decoder_config)
    print('load encoder param')

    param_dict_encoder = load_checkpoint(cfg.encoder_checkpoint_dir)
    load_param_into_net(encoder, param_dict_encoder)

    print('load decoder param')
    param_dict_decoder = load_checkpoint(cfg.decoder_checkpoint_dir)
    load_param_into_net(decoder, param_dict_decoder)

    roberta_shared = EncoderDecoderModel(
        encoder=encoder, decoder=decoder, tie_encoder_decoder=True)

    # define loss
    if cfg.label_smoothing == 0:
        loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    else:
        loss_fn = LabelSmoothedNllLoss()

    # build loss network
    loss_net = EncoderDecoderWithLossCell(
        roberta_shared, loss_fn, cfg.pad_token_id, cfg.label_smoothing)

    rank_size, rank_id = _get_rank_info()

    train_data = create_dataset(cfg.batch_size, data_file_path=cfg.train_data_dir, rank_size=rank_size,
                                rank_id=rank_id, do_shuffle=True)
    steps_per_epoch = train_data.get_dataset_size()

    if cfg.enable_dynamic_lr:

        if optimizer_cfg.optimizer == 'AdamWeightDecay':
            lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                          training_steps=steps_per_epoch * cfg.epoch,
                                          learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                          warmup_steps=optimizer_cfg.AdamWeightDecay.warmup_steps,
                                          hidden_size=cfg.hidden_size,
                                          start_decay_step=optimizer_cfg.AdamWeightDecay.start_decay_step,
                                          min_lr=optimizer_cfg.AdamWeightDecay.end_learning_rate), mstype.float32)
            params = loss_net.trainable_params()
            decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
            other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
            group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                            {'params': other_params, 'weight_decay': 0.0}]

            opt = AdamWeightDecay(group_params, lr, eps=optimizer_cfg.AdamWeightDecay.eps)
        elif optimizer_cfg.optimizer == 'Adam':
            lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                          training_steps=steps_per_epoch * cfg.epoch,
                                          learning_rate=optimizer_cfg.Adam.learning_rate,
                                          warmup_steps=optimizer_cfg.Adam.warmup_steps,
                                          hidden_size=cfg.hidden_size,
                                          start_decay_step=optimizer_cfg.Adam.start_decay_step,
                                          min_lr=optimizer_cfg.Adam.end_learning_rate), mstype.float32)
            opt = nn.Adam(loss_net.trainable_params(), learning_rate=lr)
    else:
        lr = cfg.lr
        opt = nn.Adam(roberta_shared.trainable_params(), beta1=0.9, beta2=0.98, learning_rate=lr)

    if cfg.enable_lossscale:
        manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value ** 32,
                                                scale_factor=cfg.scale_factor, scale_window=cfg.scale_window)
        train_model = nn.TrainOneStepWithLossScaleCell(
            loss_net, opt, scale_sense=manager)
        model = Model(train_model)
    else:
        model = Model(network=loss_net, optimizer=opt)
    loss_cb = LossCallBack(rank_id=0, per_print_times=10)

    time_cb = TimeMonitor(train_data.get_dataset_size())
    callbacks = [loss_cb, time_cb]

    if cfg.device_num == 1 or (cfg.device_num > 1 and rank_id == 0):
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                       keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='roberta_seq2seq', directory=save_ckpt_path, config=ckpt_config)
        callbacks.append(ckpoint_cb)

    begin_time = time.time()
    #print_times = 10
    #epochs = int(cfg.epoch * train_data.get_dataset_size() / print_times)
    # model.train(epoch=epochs, train_dataset=train_data, callbacks=callbacks,
    #             dataset_sink_mode=True, sink_size=print_times)
    model.train(epoch=cfg.epoch, train_dataset=train_data, callbacks=callbacks, dataset_sink_mode=False)
    save_checkpoint(loss_net, os.path.join(save_ckpt_path, "roberta_seq2seq_last.ckpt"))
    end_time = time.time()
    run_time = end_time - begin_time
    print('train time：', run_time)
    print('train over')


if __name__ == '__main__':
    run_train()
