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
"""PWCNet train."""
import os
import time
import datetime
import warnings

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, RunContext, CheckpointConfig
from mindspore.nn import Adam
from mindspore.communication.management import get_group_size, init, get_rank
from mindspore.nn import TrainOneStepCell

from src.pwcnet_model import PWCNet, BuildTrainNetwork
from src.lr_generator import MultiStepLR
from src.sintel import SintelTraining
from src.flyingchairs import FlyingChairsTrain
from src.log import get_logger, AverageMeter
from src.loss import MultiScaleEPE_PWC

from model_utils.config import config as cfg
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

warnings.filterwarnings('ignore')
mindspore.common.seed.set_seed(1)

cfg.lr_epochs = list(map(int, cfg.lr_epochs.split(',')))

class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, _key):
        return self[_key]

    def __setattr__(self, _key, _value):
        self[_key] = _value

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, cfg.modelarts_dataset_unzip_name)):
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

    if cfg.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(cfg.data_url, cfg.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(cfg.data_path, cfg.modelarts_dataset_unzip_name)
        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    cfg.ckpt_path = os.path.join(cfg.checkpoint_url)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train(de_dataloader, de_dataloader_val):
    '''run_train'''
    cfg.logger.important_info('start create network')
    create_network_start = time.time()
    network = PWCNet()
    criterion = MultiScaleEPE_PWC()

    # load pretrain model
    if os.path.isfile(cfg.pretrained):
        param_dict = load_checkpoint(cfg.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moment1.' or 'moment2' or 'global_step' or 'beta1_power' or 'beta2_power' or
                              'learning_rate'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        cfg.logger.info('load model %s success.', cfg.pretrained)

    # optimizer and lr scheduler
    lr_scheduler = MultiStepLR(cfg.lr,
                               cfg.lr_epochs,
                               cfg.lr_gamma,
                               cfg.steps_per_epoch,
                               cfg.max_epoch,
                               warmup_epochs=cfg.warmup_epochs)
    lr_schedule = lr_scheduler.get_lr()
    opt = Adam(params=network.trainable_params(), learning_rate=Tensor(lr_schedule), loss_scale=cfg.loss_scale)
    # package training process, adjust lr + forward + backward + optimizer
    train_net = BuildTrainNetwork(network, criterion)
    # TrainOneStepCell api changed since TR5_branch 2020/03/09
    train_net = TrainOneStepCell(train_net, opt, sens=cfg.loss_scale)

    # checkpoint save
    if cfg.local_rank == 0:
        ckpt_max_num = cfg.max_epoch * cfg.steps_per_epoch // cfg.ckpt_interval
        train_config = CheckpointConfig(save_checkpoint_steps=cfg.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=train_config, directory=cfg.outputs_dir, prefix='{}'.format(cfg.local_rank))
        cb_params = InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    train_net.set_train()
    t_end = time.time()
    t_epoch = time.time()
    old_progress = -1
    loss_meter = AverageMeter('loss')

    cfg.logger.important_info('====start train====')
    for i, data in enumerate(de_dataloader):
        # clean grad + adjust lr + put data into device + forward + backward + optimizer, return loss
        loss = train_net(data[0], data[1], data[2])
        loss_meter.update(loss.asnumpy())

        if cfg.local_rank == 0:
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        # logging loss, fps, ...
        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            cfg.logger.important_info('{}, graph compile time={:.2f}s'.format(cfg.task, time_for_graph_compile))

        if i % cfg.log_interval == 0 and cfg.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / cfg.steps_per_epoch)
            fps = cfg.batch_size * (i - old_progress) * cfg.world_size / time_used
            cfg.logger.info('epoch[{}], iter[{}], {:.2f} imgs/sec\t Loss '
                            '{loss_meter.val:.4f} {loss_meter.avg:.4f}'.format(epoch, i, fps, loss_meter=loss_meter))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i


        if i % cfg.steps_per_epoch == 0 and cfg.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / cfg.steps_per_epoch)
            fps = cfg.batch_size * cfg.world_size * cfg.steps_per_epoch / epoch_time_used
            cfg.logger.info('=================================================')
            cfg.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            t_epoch = time.time()
            validation_loss = 0
            sum_num = 0
            for _, val_data in enumerate(de_dataloader_val):
                network.set_train(False)
                val_output = network(val_data[0], val_data[1], training=False)
                val_loss = criterion(val_output, val_data[2], training=False)
                validation_loss += val_loss
                sum_num += 1
            print('validation EPE: ', validation_loss / sum_num)
    cfg.logger.important_info('====train end====')


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    if cfg.device_target == 'Ascend':
        context.set_context(device_id=get_device_id())

    # Init distributed
    if cfg.is_distributed:
        init()
        cfg.local_rank = get_rank()
        cfg.world_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        parallel_mode = ParallelMode.STAND_ALONE

    # parallel_mode 'STAND_ALONE' do not support parameter_broadcast and mirror_mean
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=cfg.world_size, gradients_mean=True)
    cfg.outputs_dir = os.path.join(cfg.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    cfg.logger = get_logger(cfg.outputs_dir, cfg.local_rank)

    # Dataloader
    de_ds, _ = FlyingChairsTrain(cfg.train_label_file, cfg.training_augmentations, 'train', cfg.batch_size,
                                 cfg.num_parallel_workers, cfg.local_rank, cfg.world_size)
    cfg.steps_per_epoch = de_ds.get_dataset_size()
    de_ds = de_ds.repeat(cfg.max_epoch)
    de_ds_val, _ = SintelTraining(cfg.eval_dir, cfg.valid_augmentations, 'final', 'valid', cfg.val_batch_size,
                                  cfg.num_parallel_workers, 0, 1)
    de_ds = de_ds.create_tuple_iterator(output_numpy=False, do_copy=False)
    de_ds_val = de_ds_val.create_tuple_iterator(output_numpy=False, do_copy=False)
    cfg.logger.save_args(cfg)

    run_train(de_ds, de_ds_val)
