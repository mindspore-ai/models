# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""training script"""

import os
import time
import socket

import glob
import numpy as np
import moxing as mox
import mindspore as ms
from mindspore import Tensor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
from mindspore.profiler.profiling import Profiler
import mindspore.dataset as ds

from src.vit import get_network
from src.dataset import get_dataset
from src.cross_entropy import get_loss
from src.optimizer import get_optimizer
from src.lr_generator import get_lr
from src.eval_engine import get_eval_engine
from src.callback import StateMonitor
from src.logging import get_logger
from src.model_utils.config import config



try:
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')

    device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
    local_rank = int(os.getenv('RANK_ID'))    # local_rank
    device_num = int(os.getenv('RANK_SIZE'))  # world_size
    print("distribute training")
except TypeError:
    device_id = 0   # 0 ~ 7
    local_rank = 0    # local_rank
    device_num = 1  # world_size
    print("standalone training")

def add_static_args(args):
    """add_static_args"""
    args.weight_decay = float(args.weight_decay)
    args.eval_engine = 'imagenet'
    args.split_point = 0.4
    args.poly_power = 2
    args.aux_factor = 0.4
    args.seed = 1
    args.auto_tune = 0

    if args.eval_offset < 0:
        args.eval_offset = args.max_epoch % args.eval_interval

    args.device_id = device_id
    args.local_rank = local_rank
    args.device_num = device_num
    return args

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def frozen_to_air(network, args):
    param_dict_t = ms.load_checkpoint(args.get("ckpt_file"))
    ms.load_param_into_net(network, param_dict_t)
    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[args.get("batch_size"), 3, args.get("width"), \
        args.get("height")]), ms.float32)
    ms.export(network, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))

if __name__ == '__main__':

    args_opt = add_static_args(config)
    np.random.seed(args_opt.seed)
    args_opt.logger = get_logger(config.output_path, rank=local_rank)

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path, exist_ok=True)
    if not os.path.exists(config.load_path):
        os.makedirs(config.load_path, exist_ok=True)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)
    mox.file.copy_parallel(config.data_url, config.data_path)
    if args_opt.enable_transfer:
        mox.file.copy_parallel(config.pretrained_url, config.load_path)
        ckpt_name = os.listdir(config.load_path)
        ckpt_name = ckpt_name[-1]
        config.ckpt_path = os.path.join(config.load_path, ckpt_name)

    config.epoch_size = config.epoch_size
    config.batch_size = config.batch_size
    config.dataset_path = os.path.join(config.data_path, "train")

    ms.set_context(device_id=device_id,
                   mode=ms.GRAPH_MODE,
                   device_target="Ascend",
                   save_graphs=False)
    if args_opt.auto_tune:
        ms.set_context(auto_tune_mode='GA')
    elif args_opt.device_num == 1:
        pass
    else:
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode=ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)

    if args_opt.open_profiler:
        profiler = Profiler(output_path="data_{}".format(local_rank))

    # init the distribute env
    if not args_opt.auto_tune and args_opt.device_num > 1:
        init()

    # network
    net = get_network(backbone_name=args_opt.backbone, args=args_opt)

    # set grad allreduce split point
    parameters = [param for param in net.trainable_params()]
    parameter_len = len(parameters)
    if args_opt.split_point > 0:
        print("split_point={}".format(args_opt.split_point))
        split_parameter_index = [int(args_opt.split_point*parameter_len),]
        parameter_indices = 1
        for i in range(parameter_len):
            if i in split_parameter_index:
                parameter_indices += 1
            parameters[i].comm_fusion = parameter_indices
    else:
        print("warning!!!, no split point")

    if os.path.isfile(config.ckpt_path):
        ckpt = ms.load_checkpoint(config.ckpt_path)
        filter_list = [x.name for x in net.head.get_parameters()]
        filter_checkpoint_parameter_by_list(ckpt, filter_list)
        ms.load_param_into_net(net, ckpt)

    # loss
    if not args_opt.use_label_smooth:
        args_opt.label_smooth_factor = 0.0
    loss = get_loss(loss_name=args_opt.loss_name, args=args_opt)

    # train dataset
    epoch_size = config.epoch_size
    dataset = get_dataset(dataset_name=args_opt.dataset_name,
                          do_train=True,
                          dataset_path=config.dataset_path,
                          args=args_opt)
    ds.config.set_seed(args_opt.seed)
    step_size = dataset.get_dataset_size()
    args_opt.steps_per_epoch = step_size
    # evaluation dataset
    eval_dataset = get_dataset(dataset_name=args_opt.dataset_name,
                               do_train=False,
                               dataset_path=config.dataset_path,
                               args=args_opt)

    # evaluation engine
    if args_opt.auto_tune or args_opt.open_profiler or eval_dataset is None or args_opt.device_num == 1:
        args_opt.eval_engine = ''
    eval_engine = get_eval_engine(args_opt.eval_engine, net, eval_dataset, args_opt)

    # loss scale
    loss_scale = FixedLossScaleManager(args_opt.loss_scale, drop_overflow_update=False)

    # learning rate
    lr_array = get_lr(global_step=0, lr_init=args_opt.lr_init, lr_end=args_opt.lr_min, lr_max=args_opt.lr_max,
                      warmup_epochs=args_opt.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                      lr_decay_mode=args_opt.lr_decay_mode, poly_power=args_opt.poly_power)
    lr = Tensor(lr_array)

    # optimizer, group_params used in grad freeze
    opt, _ = get_optimizer(optimizer_name=args_opt.opt,
                           network=net,
                           lrs=lr,
                           args=args_opt)

    # model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  metrics=eval_engine.metric, eval_network=eval_engine.eval_network,
                  loss_scale_manager=loss_scale, amp_level="O3")
    eval_engine.set_model(model)
    args_opt.logger.save_args(args_opt)

    t0 = time.time()
    # equal to model._init(dataset, sink_size=step_size)
    eval_engine.compile(sink_size=step_size)

    t1 = time.time()
    args_opt.logger.info('compile time used={:.2f}s'.format(t1 - t0))

    # callbacks
    state_cb = StateMonitor(data_size=step_size,
                            tot_batch_size=config.batch_size * device_num,
                            lrs=lr_array,
                            eval_interval=args_opt.eval_interval,
                            eval_offset=args_opt.eval_offset,
                            eval_engine=eval_engine,
                            logger=args_opt.logger.info)

    cb = [state_cb,]
    if args_opt.save_checkpoint and local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=1,
                                     keep_checkpoint_max=5,
                                     async_save=True)
        ckpt_cb = ModelCheckpoint(prefix=args_opt.backbone, directory=config.output_path, config=config_ck)
        cb += [ckpt_cb]

    t0 = time.time()
    model.train(epoch_size, dataset, callbacks=cb, sink_size=step_size)
    t1 = time.time()
    args_opt.logger.info('training time used={:.2f}s'.format(t1 - t0))
    last_metric = 'last_metric[{}]'.format(state_cb.best_acc)
    args_opt.logger.info(last_metric)

    is_cloud = args_opt.enable_modelarts
    if is_cloud:
        ip = os.getenv("BATCH_TASK_CURRENT_HOST_IP")
    else:
        ip = socket.gethostbyname(socket.gethostname())
    args_opt.logger.info('ip[{}], mean_fps[{:.2f}]'.format(ip, state_cb.mean_fps))

    if args_opt.open_profiler:
        profiler.analyse()

    net = get_network(backbone_name=args_opt.backbone, args=args_opt)


    ckpt_list = glob.glob(config.output_path + "/vit*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    frozen_to_air_args = {'ckpt_file': ckpt_model, 'batch_size': 1, 'height': 224, 'width': 224,
                          'file_name': config.output_path + '/VIT', 'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)
    mox.file.copy_parallel(config.output_path, config.train_url)
    print("train success")
