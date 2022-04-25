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
""" Train script """

import os
import time

from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from model_utils.config import get_config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.callbacks import SavingLossMonitor, SavingTimeMonitor
from src.dataset import create_dataset
from src.loss import StrongLoss
from src.lr_schedule import get_step_lr
from src.model.strong_reid import ReIDStrong
from src.model.cell_wrapper import TrainOneStepCell, NetworkWithCell

set_seed(1)
config = get_config()


def modelarts_pre_process():
    """ Modelarts pre process function """
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


def _prepare_configuration():
    """Prepare configuration"""
    config.image_size = list(map(int, config.image_size.split(',')))
    config.image_mean = list(map(float, config.image_mean.split(',')))
    config.image_std = list(map(float, config.image_std.split(',')))

    _enable_graph_kernel = False
    context.set_context(
        mode=context.GRAPH_MODE,
        enable_graph_kernel=_enable_graph_kernel,
        device_target=config.device_target,
    )

    config.device_id = get_device_id()

    if config.is_distributed:
        if config.device_target == "GPU":
            if not config.enable_modelarts:
                init()
            else:
                if not config.need_modelarts_dataset_unzip:
                    init()

            config.group_size = get_group_size()
            config.rank = get_rank()

            device_num = config.group_size
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
        if config.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            config.rank = get_rank()
            config.group_size = get_group_size()
    else:
        config.group_size = 1
        config.rank = 0

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # select for master rank printing logs or all rank save, compatible for model parallel
    config.rank_print_ckpt_flag = 0
    if config.is_print_on_master:
        if config.rank == 0:
            config.rank_print_ckpt_flag = 1
    else:
        config.rank_print_ckpt_flag = 1


def get_optimizers(network, criterion, batch_num):
    """ Create Adam optimizer for network and SGD optimizer for Center loss """
    lr = get_step_lr(
        lr_init=config.lr_init,
        total_epochs=config.max_epoch,
        steps_per_epoch=batch_num,
        decay_epochs=config.decay_epochs,
        warmup_epoch=config.warmup_epoch,
        warmup_factor=config.warmup_factor,
    )
    lr = Tensor(lr)
    adam_group_params = [
        {"params": network.trainable_params(), 'weight_decay': config.weight_decay},
    ]
    opt_net = nn.Adam(adam_group_params, learning_rate=lr)
    for name, par in opt_net.parameters_dict().items():
        if '.' not in name:
            par.name = 'adam.'+name
    group_params = [
        {"params": criterion.trainable_params(), "lr": config.lr_cri},
    ]
    opt_cri = nn.SGD(group_params)

    return opt_net, opt_cri


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """ Run train """
    _prepare_configuration()

    dataset = create_dataset(
        config.data_dir,
        ims_per_id=config.ims_per_id,
        ids_per_batch=config.ids_per_batch,
        mean=config.image_mean,
        std=config.image_std,
        resize_h_w=config.image_size,
        padding=config.padding,
        rank=config.rank,
        group_size=config.group_size,
        dataset_name=config.dataset,
    )

    batch_num = dataset.get_dataset_size()
    config.batch_size = dataset.batch_size

    network = ReIDStrong(num_classes=config.n_classes, pretrained_backbone=config.pre_trained_backbone)

    # pre_trained
    if config.pre_trained:
        print('Load model from', config.pre_trained)
        load_param_into_net(network, load_checkpoint(config.pre_trained))

    reid_loss = StrongLoss(
        config.n_classes,
        config.center_loss_weight,
        config.crossentropy_loss_weight,
        margin=config.global_loss_margin,
    )

    optimizers = get_optimizers(network, reid_loss, batch_num)

    timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + str(config.rank)

    logfile = SavingTimeMonitor.open_file(
        config.train_log_path if config.rank_save_ckpt_flag else None,
        timestamp=timestamp,
    )

    time_cb = SavingTimeMonitor(data_size=batch_num, logfile=logfile)

    callbacks = [time_cb]
    if config.log_interval is None:
        config.log_interval = batch_num

    if config.rank_print_ckpt_flag:

        loss_cb = SavingLossMonitor(
            per_print_times=config.log_interval,
            logfile=logfile,
            init_info=str(config),
        )
        callbacks.append(loss_cb)

    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=int(config.ckpt_interval * batch_num),
            keep_checkpoint_max=config.keep_checkpoint_max,
        )
        save_ckpt_path = os.path.join(config.ckpt_path, timestamp + '-ckpt' + '/')
        ckpt_cb = ModelCheckpoint(
            config=ckpt_config,
            directory=save_ckpt_path,
            prefix='{}'.format(config.rank),
        )
        callbacks.append(ckpt_cb)

    network_loss = NetworkWithCell(network, reid_loss)
    network_loss = TrainOneStepCell(network_loss, optimizers, center_loss_weight=config.center_loss_weight)
    model = Model(network_loss)

    model.train(config.max_epoch, dataset, callbacks=callbacks, dataset_sink_mode=False)


if __name__ == '__main__':
    run_train()
