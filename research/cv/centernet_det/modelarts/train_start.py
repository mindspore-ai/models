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
Train CenterNet and get network model files(.ckpt)
"""

import os
import sys
import subprocess
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Adam
from mindspore import log as logger
from mindspore.common import set_seed
from mindspore.profiler import Profiler

try:
    from src.dataset import COCOHP
    from src.centernet_det import CenterNetLossCell, CenterNetWithLossScaleCell
    from src.centernet_det import CenterNetWithoutLossScaleCell
    from src.utils import LossCallBack, CenterNetPolynomialDecayLR, CenterNetMultiEpochsDecayLR
    from src.model_utils.config import config, dataset_config, net_config, train_config, export_config
    from src.model_utils.moxing_adapter import moxing_wrapper
    from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

except ImportError as import_error:
    print('Import Error: {}, trying append path/centernet_det/src/../'.format(import_error))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from src.dataset import COCOHP
    from src.centernet_det import CenterNetLossCell, CenterNetWithLossScaleCell
    from src.centernet_det import CenterNetWithoutLossScaleCell
    from src.utils import LossCallBack, CenterNetPolynomialDecayLR, CenterNetMultiEpochsDecayLR
    from src.model_utils.config import config, dataset_config, net_config, train_config, export_config
    from src.model_utils.moxing_adapter import moxing_wrapper
    from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num


def _set_parallel_all_reduce_split():
    """set centernet all_reduce fusion split"""
    context.set_auto_parallel_context(all_reduce_fusion_config=[18, 59, 100, 141, 182])


def _get_params_groups(network, optimizer):
    """get param groups"""
    params = network.trainable_params()
    decay_params = list(filter(lambda x: not optimizer.decay_filter(x), params))
    other_params = list(filter(optimizer.decay_filter, params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]
    return group_params


def _get_optimizer(network, dataset_size):
    """get optimizer, only support Adam right now."""
    if train_config.optimizer == 'Adam':
        group_params = _get_params_groups(network, train_config.Adam)
        if train_config.lr_schedule == "PolyDecay":
            lr_schedule = CenterNetPolynomialDecayLR(learning_rate=train_config.PolyDecay.learning_rate,
                                                     end_learning_rate=train_config.PolyDecay.end_learning_rate,
                                                     warmup_steps=train_config.PolyDecay.warmup_steps,
                                                     decay_steps=config.train_steps,
                                                     power=train_config.PolyDecay.power)
            optimizer = Adam(group_params, learning_rate=lr_schedule, eps=train_config.PolyDecay.eps, loss_scale=1.0)
        elif train_config.lr_schedule == "MultiDecay":
            multi_epochs = train_config.MultiDecay.multi_epochs
            if not isinstance(multi_epochs, (list, tuple)):
                raise TypeError("multi_epochs must be list or tuple.")
            if not multi_epochs:
                multi_epochs = [config.epoch_size]
            lr_schedule = CenterNetMultiEpochsDecayLR(learning_rate=train_config.MultiDecay.learning_rate,
                                                      warmup_steps=train_config.MultiDecay.warmup_steps,
                                                      multi_epochs=multi_epochs,
                                                      steps_per_epoch=dataset_size,
                                                      factor=train_config.MultiDecay.factor)
            optimizer = Adam(group_params, learning_rate=lr_schedule, eps=train_config.MultiDecay.eps, loss_scale=1.0)
        else:
            raise ValueError("Don't support lr_schedule {}, only support [PolynormialDecay, MultiEpochDecay]".
                             format(train_config.optimizer))
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, Adam]".
                         format(train_config.optimizer))
    return optimizer


def _get_last_ckpt(ckpt_dir):
    """get ckpt data"""
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_dir = os.path.join(ckpt_dir, file_dict[max_ctime])
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    print("ckpt file name:", sorted(ckpt_files)[-1])
    if not ckpt_files:
        print("No ckpt file found.")
        return None
    ckpt_file_last = os.path.join(ckpt_dir, sorted(ckpt_files)[-1])
    return ckpt_file_last, ckpt_dir


def _export_air(ckpt_dir):
    """convert ckpt to air"""
    ckpt_file, ckpt_dir = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        print('Freezing model failed!')
        print("can not find ckpt files. ")
        return
    file_name = os.path.join(ckpt_dir, export_config.export_name)
    export_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, "export.py")
    cmd = ["python", export_file,
           f"--device_id={config.device_id}",
           f"--export_format={export_config.export_format}",
           f"--export_load_ckpt={ckpt_file}",
           f"--export_name={file_name}",
           f"--num_classes={config.num_classes}"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def modelarts_pre_process():
    """modelarts pre process function."""
    config.mindrecord_dir = config.data_path
    config.save_checkpoint_path = os.path.join(config.output_path, config.save_checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    """training CenterNet"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(reserve_class_name_in_scope=False)
    context.set_context(save_graphs=False)

    ckpt_save_dir = config.save_checkpoint_path
    rank = 0
    device_num = 1
    num_workers = 8

    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())
        if config.distribute == "true":

            D.init()
            device_num = get_device_num()
            rank = get_rank_id()
            ckpt_save_dir = config.save_checkpoint_path + 'ckpt_' + str(get_rank()) + '/'

            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            _set_parallel_all_reduce_split()
    else:
        config.distribute = "false"
        config.need_profiler = "false"
        config.enable_data_sink = "false"

    # Start create dataset!
    # mindrecord files will be generated at args_opt.mindrecord_dir such as centernet.mindrecord0, 1, ... file_num.
    logger.info("Begin creating dataset for CenterNet")
    coco = COCOHP(dataset_config, run_mode="train", net_opt=net_config, save_path=config.save_result_dir)
    dataset = coco.create_train_dataset(config.mindrecord_dir, config.mindrecord_prefix,
                                        batch_size=train_config.batch_size, device_num=device_num, rank=rank,
                                        num_parallel_workers=num_workers, do_shuffle=config.do_shuffle == 'true')
    dataset_size = dataset.get_dataset_size()
    logger.info("Create dataset done!")

    net_with_loss = CenterNetLossCell(net_config)

    config.train_steps = config.epoch_size * dataset_size
    logger.info("train steps: {}".format(config.train_steps))

    optimizer = _get_optimizer(net_with_loss, dataset_size)

    enable_static_time = config.device_target == "CPU"
    callback = [TimeMonitor(config.data_sink_steps), LossCallBack(dataset_size, enable_static_time)]
    if config.enable_save_ckpt == "true" and get_device_id() % min(8, device_num) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_centernet',
                                     directory=None if ckpt_save_dir == "" else ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)

    if config.load_checkpoint_path:
        param_dict = load_checkpoint(config.load_checkpoint_path)
        if config.filter_weight:
            for key in list(param_dict.keys()):
                if 'hm_fn' in key:
                    print("Delete parameter from checkpoint: ", key)
                    del param_dict[key]
        load_param_into_net(net_with_loss, param_dict)

    if config.device_target == "Ascend":
        net_with_grads = CenterNetWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                    sens=train_config.loss_scale_value)
    else:
        net_with_grads = CenterNetWithoutLossScaleCell(net_with_loss, optimizer=optimizer)

    model = Model(net_with_grads)
    model.train(config.epoch_size, dataset, callbacks=callback,
                dataset_sink_mode=(config.enable_data_sink == "true"), sink_size=config.data_sink_steps)


def main():
    train()
    print('CenterNet training success!')
    import moxing as mox
    _export_air(config.output_path)
    mox.file.copy_parallel(config.output_path, config.train_url)
    print("Finish sync data from {} to {}.".format(config.output_path, config.train_url))
    return 0


if __name__ == '__main__':
    if config.need_profiler == "true":
        profiler = Profiler(output_path=config.profiler_path)
    set_seed(317)
    main()
    if config.need_profiler == "true":
        profiler.analyse()
