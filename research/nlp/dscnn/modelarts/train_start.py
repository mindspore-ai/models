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
"""DSCNN train."""
import os
import datetime
import argparse
import ast
import numpy as np
from mindspore import context
from mindspore import Tensor, Model
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, export
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, TimeMonitor, CheckpointConfig
from src.log import get_logger
from src.dataset import audio_dataset
from src.ds_cnn import DSCNN
from src.loss import CrossEntropy
from src.models import load_ckpt
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.callback import ProgressMonitor
from src.model_utils.config import parse_yaml, merge, prepare_words_list, Config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num


current_path = current_path = os.path.dirname(os.path.realpath(__file__)) # BootfileDirectory, 启动文件所在的目录
config_path = os.path.join(current_path, 'default_config.yaml')

def _parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path='default_config.yaml'):
    """
    Parse command line arguments to the configuration according to the default yaml

    Args:
        parser: Parent parser
        cfg: Base configuration
        helper: Helper description
        cfg_path: Path to the default yaml config
    """
    parser = argparse.ArgumentParser(description='[REPLACE THIS at config.py]',
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if item in parser.parse_args():
            continue
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else 'Please reference to {}'.format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument('--' + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument('--' + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args

def _get_config():
    """
    Get Config according to the yaml file and cli arguments
    """
    parser = argparse.ArgumentParser(description='default name', add_help=False)
    parser.add_argument('--config_path', type=str, default=os.path.join(config_path),
                        help='Config file path')
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--amp_level', default='O0')

    # dataset
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument('--per_batch_size', type=int, default=100,
                        help='batch size')

    # optimizer
    parser.add_argument('--max_epoch', type=int, default=80, help='epoch')
    parser.add_argument('--lr_scheduler', type=str, default='multistep',
                        help='type of learning rate')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='base learning rate')

    # model
    parser.add_argument('--pretrained', type=str, default='',
                        help='pretrained model to load')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--is_distributed', action='store_true', help='distributed training')
    parser.add_argument('--file_name', type=str, default='dscnn', help='CNN&CTC output air name')

    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = _parse_cli_to_yaml(parser=parser, cfg=default, helper=helper,
                              choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)

    configs = Config(final_config)
    configs.dataset_sink_mode = bool(configs.use_graph_mode)
    configs.lr_epochs = list(map(int, configs.lr_epochs.split(',')))
    configs.model_setting_dropout1 = configs.drop

    configs.model_setting_desired_samples = int(configs.sample_rate * configs.clip_duration_ms / 1000)
    configs.model_setting_window_size_samples = int(configs.sample_rate * configs.window_size_ms / 1000)
    configs.model_setting_window_stride_samples = int(configs.sample_rate * configs.window_stride_ms / 1000)
    length_minus_window = (configs.model_setting_desired_samples - configs.model_setting_window_size_samples)

    if length_minus_window < 0:
        configs.model_setting_spectrogram_length = 0
    else:
        configs.model_setting_spectrogram_length = 1 + int(length_minus_window
                                                           / configs.model_setting_window_stride_samples)

    configs.model_setting_fingerprint_size = configs.dct_coefficient_count * configs.model_setting_spectrogram_length
    configs.model_setting_label_count = len(prepare_words_list(configs.wanted_words.split(',')))
    configs.model_setting_sample_rate = configs.sample_rate
    configs.model_setting_dct_coefficient_count = configs.dct_coefficient_count

    return configs

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def val(args, model, val_dataset):
    '''Eval.'''
    val_dataloader = val_dataset.create_tuple_iterator()
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    if args.amp_level == 'O0':
        origin_mstype = mstype.float32
    else:
        origin_mstype = mstype.float16
    model.predict_network.to_float(mstype.float32)

    for data, gt_classes in val_dataloader:
        output = model.predict(Tensor(data, mstype.float32))
        output = output.asnumpy()
        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[:, -5:]
        gt_classes = gt_classes.asnumpy()
        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += output.shape[0]

    model.predict_network.to_float(origin_mstype)
    results = [[top1_correct], [top5_correct], [img_tot]]

    results = np.array(results)

    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    acc5 = 100.0 * top5_correct / img_tot
    if acc1 > args.best_acc:
        args.best_acc = acc1
        args.best_epoch = args.epoch_cnt - 1
    args.logger.info('Eval: top1_cor:{}, top5_cor:{}, tot:{}, acc@1={:.2f}%, acc@5={:.2f}%' \
                     .format(top1_correct, top5_correct, img_tot, acc1, acc5))

def _callback_func(args, cb, prefix):
    callbacks = [cb]
    if args.rank_save_ckpt_flag:
        ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=args.outputs_dir, prefix=prefix)
        callbacks.append(ckpt_cb)
    callbacks.append(TimeMonitor(args.per_batch_size))
    return callbacks

def trainval(args, model, train_dataset, val_dataset, cb, rank):
    callbacks = _callback_func(args, cb, 'epoch{}'.format(args.epoch_cnt))
    model.train(args.val_interval, train_dataset, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)
    if rank == 0:
        val(args, model, val_dataset)


def modelarts_pre_process():
    pass


def _get_last_ckpt(args, ckpt_dir):
    '''Get the ckpt file'''
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_dir = os.path.join(ckpt_dir, file_dict[max_ctime])
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None
    prefix_continue = True
    postfix_continue = True
    prefix_index = 0
    postfix_index = -1
    prefix = ''
    postfix = ''
    while prefix_continue or postfix_continue:
        if prefix_continue:
            tmp = prefix + ckpt_files[0][prefix_index]
            prefix_index = prefix_index + 1
            res = []
            for ckpt_file in ckpt_files:
                if ckpt_file.startswith(tmp):
                    res.append(1)
                else:
                    res.append(0)
            if sum(res) == len(res):
                prefix = tmp
            else:
                prefix_continue = False
        if postfix_continue:
            tmp_list = list(postfix)
            tmp_list.insert(0, ckpt_files[0][postfix_index])
            tmp = ''.join(tmp_list)
            postfix_index = postfix_index - 1
            res = []
            for ckpt_file in ckpt_files:
                if ckpt_file.endswith(tmp):
                    res.append(1)
                else:
                    res.append(0)
            if sum(res) == len(res):
                postfix = tmp
            else:
                postfix_continue = False
    return os.path.join(ckpt_dir, prefix + str(args.best_epoch) + postfix)

def _export_air(args, ckpt_dir):
    '''Export model'''
    ckpt_file = _get_last_ckpt(args, ckpt_dir)
    if not ckpt_file:
        return

    print(f"Start exporting AIR")
    args.file_name = os.path.join(args.train_url, args.file_name)
    network = DSCNN(args, args.model_size_info)
    load_ckpt(network, ckpt_file, False)
    x = np.random.uniform(0.0, 1.0, size=[1, 1, args.model_setting_spectrogram_length,
                                          args.model_setting_dct_coefficient_count]).astype(np.float32)
    export(network, Tensor(x), file_name=args.file_name, file_format="AIR")
    export(network, Tensor(x), file_name=args.file_name, file_format="ONNX")

@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    '''Train.'''
    config = _get_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    config.rank_save_ckpt_flag = 1
    config.save_ckpt_path = config.train_url
    config.train_feat_dir = config.data_url
    # init distributed
    if config.is_distributed:
        if get_device_id():
            context.set_context(device_id=get_device_id())
        init()
        rank = get_rank_id()
        device_num = get_device_num()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=device_num, gradients_mean=True)
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=get_device_id())
    # Logger
    config.outputs_dir = os.path.join(config.save_ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir)

    # Dataloader: train, val
    train_dataset = audio_dataset(config.train_feat_dir, 'training', config.model_setting_spectrogram_length,
                                  config.model_setting_dct_coefficient_count, config.per_batch_size, device_num, rank)
    config.steps_per_epoch = train_dataset.get_dataset_size()
    val_dataset = audio_dataset(config.train_feat_dir, 'validation', config.model_setting_spectrogram_length,
                                config.model_setting_dct_coefficient_count, config.per_batch_size)

    # show args
    config.logger.save_args(config)

    # Network
    config.logger.important_info('start create network')
    network = DSCNN(config, config.model_size_info)

    # Load pretrain model
    if os.path.isfile(config.pretrained):
        load_checkpoint(config.pretrained, network)
        config.logger.info('load model %s success', config.pretrained)

    # Loss
    criterion = CrossEntropy(num_classes=config.model_setting_label_count)

    # LR scheduler
    if config.lr_scheduler == 'multistep':
        lr_scheduler = MultiStepLR(config.lr, config.lr_epochs, config.lr_gamma, config.steps_per_epoch,
                                   config.max_epoch, warmup_epochs=config.warmup_epochs)
    elif config.lr_scheduler == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(config.lr, config.T_max, config.steps_per_epoch, config.max_epoch,
                                         warmup_epochs=config.warmup_epochs, eta_min=config.eta_min)
    else:
        raise NotImplementedError(config.lr_scheduler)
    lr_schedule = lr_scheduler.get_lr()

    # Optimizer
    opt = Momentum(params=network.trainable_params(),
                   learning_rate=Tensor(lr_schedule),
                   momentum=config.momentum,
                   weight_decay=config.weight_decay)

    model = Model(network, loss_fn=criterion, optimizer=opt, amp_level=config.amp_level, keep_batchnorm_fp32=False)

    # Training
    config.epoch_cnt = 0
    config.best_epoch = 0
    config.best_acc = 0
    progress_cb = ProgressMonitor(config)
    while config.epoch_cnt + config.val_interval < config.max_epoch:
        trainval(config, model, train_dataset, val_dataset, progress_cb, rank)
    rest_ep = config.max_epoch - config.epoch_cnt
    if rest_ep > 0:
        trainval(config, model, train_dataset, val_dataset, progress_cb, rank)

    config.logger.info('Best epoch:{} acc:{:.2f}%'.format(config.best_epoch, config.best_acc))
    _export_air(config, config.train_url)


if __name__ == "__main__":
    train()
