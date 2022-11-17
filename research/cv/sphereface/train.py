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
"""train launch."""
import os
import time
import datetime
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from src.optimizers import get_param_groups
from src.losses.crossentropy import AngleLoss
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.utils.logging import get_logger
from src.model_utils.config import config
from src.network.spherenet import sphere20a
from src.model_utils.device_adapter import get_rank_id, get_device_id
from src.datasets.classification import classification_dataset_imagenet
set_seed(100)


class BuildTrainNetwork(nn.Cell):
    """build training network"""

    def __init__(self, net, crit):
        super(BuildTrainNetwork, self).__init__()
        self.network = net
        self.criterion = crit

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


class ProgressMonitor(Callback):
    """monitor loss and time"""

    def __init__(self, configs):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.configs = configs
        self.ckpt_history = []

    def begin(self, run_context):
        self.configs.logger.info('start network train...')

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        """process epoch end"""
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.configs.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = self.configs.per_batch_size * (me_step - self.me_epoch_start_step_num) * \
                   self.configs.group_size / time_used
        self.configs.logger.info('epoch[{}], iter[{}], loss:{}, mean_fps:{:.2f} imgs/sec'.format(real_epoch
                                                                                                 , me_step,
                                                                                                 cb_params.net_outputs,
                                                                                                 fps_mean))
        if self.configs.rank_save_ckpt_flag:
            import glob
            ckpts = glob.glob(os.path.join(self.configs.outputs_dir, '*.ckpt'))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith('{}-'.format(self.configs.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.configs.logger.info('epoch[{}], iter[{}], loss:{}, ckpt:{},'
                                         'ckpt_fn:{}'.format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn))

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.configs.logger.info('end network train...')


def get_lr_scheduler(configs):
    if configs.lr_scheduler == 'exponential':
        lr_scheduler = MultiStepLR(configs.lr, configs.lr_epochs, configs.lr_gamma
                                   , configs.steps_per_epoch, configs.max_epoch, warmup_epochs=configs.warmup_epochs)
    elif config.lr_scheduler == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(configs.lr, configs.T_max, configs.steps_per_epoch, configs.max_epoch,
                                         warmup_epochs=configs.warmup_epochs, eta_min=configs.eta_min)
    else:
        raise NotImplementedError(configs.lr_scheduler)
    return lr_scheduler

def train():
    """training process"""
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.image_size = list(map(int, config.image_size.split(',')))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False)
    # init distributed set
    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    parallel_mode = ParallelMode.STAND_ALONE
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        init()
        config.group_size = get_group_size()
        config.rank = get_rank()
    if config.device_target == 'Ascend':
        devid = get_device_id()
        context.set_context(device_id=devid)
        config.rank = get_rank_id()

    # init loss_scale set
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.save_ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)

    # dataloader
    data_dir = config.train_data_dir
    images_dir = config.train_img_dir
    if config.enable_modelarts:
        import moxing as mox
        images_dir = '/cache/dataset/device' + os.getenv("DEVICE_ID")
        os.system('mkdir %s' %images_dir)
        mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset/device'+os.getenv('DEVICE_ID'))
        os.system('cd %s ; tar -xvf CASIA-WebFace.tar' % (images_dir))
        os.system('cd %s ; tar -xvf CASIA-WebFace2.tar' % (images_dir))
        images_dir = images_dir +'/'
        data_dir = images_dir + 'casia_landmark.txt'
    de_dataset = classification_dataset_imagenet(data_dir, image_size=[112, 96],
                                                 per_batch_size=config.per_batch_size, max_epoch=config.max_epoch,
                                                 rank=config.rank, group_size=config.group_size,
                                                 input_mode="txt", root=images_dir, shuffle=True)
    config.steps_per_epoch = de_dataset.get_dataset_size()
    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')
    # get network and init'
    network = sphere20a(config.num_classes, feature=False)

    # loss
    if not config.label_smooth:
        config.smooth_factor = 0.0
    criterion = AngleLoss(classnum=config.num_classes, smooth_factor=config.smooth_factor)

    # load pretrain model
    if os.path.isfile(config.train_pretrained):
        param_dict = load_checkpoint(config.train_pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                if config.filter_weight and  key == 'network.fc6.weight':
                    continue
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values

        load_param_into_net(network, param_dict_new)
        config.logger.info('load model success')

    # lr scheduler
    lr_scheduler = get_lr_scheduler(config)
    lr_schedule = lr_scheduler.get_lr()

    # optimizer
    opt = Momentum(params=get_param_groups(network), learning_rate=Tensor(lr_schedule),
                   momentum=config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    # mixed precision training
    criterion.add_flags_recursive(fp32=True)

    # package training process, adjust lr + forward + backward + optimizer
    train_net = BuildTrainNetwork(network, criterion)

    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=config.group_size,
                                      gradients_mean=True)

    if config.device_target == 'Ascend':
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O2")
    elif config.device_target == 'GPU':
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O2")
    elif config.device_target == 'CPU':
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O0")
    else:
        raise ValueError("Unsupported device target.")

    network.fc6.to_float(mindspore.dtype.float32)
    criterion.to_float(mindspore.dtype.float32)
    # checkpoint save
    progress_cb = ProgressMonitor(config)
    callbacks = [progress_cb,]
    if config.rank_save_ckpt_flag:
        ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch,
                                       keep_checkpoint_max=ckpt_max_num)
        if config.enable_modelarts:
            ckpt_cb = ModelCheckpoint(config=ckpt_config, directory='/cache/train_output/device'+os.getenv('DEVICE_ID'),
                                      prefix='%s' % config.rank)
        else:
            ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=config.save_ckpt_path,
                                      prefix='%s' % config.rank)
        callbacks.append(ckpt_cb)
    model.train(config.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=False)
    if config.enable_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)

if __name__ == "__main__":
    train()
