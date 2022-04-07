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
# =======================================================================================
""" Train entrance module """
import os
import time
import datetime
import argparse

from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore.nn import Momentum
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import context, Model, DynamicLossScaleManager, load_checkpoint, load_param_into_net
from mindspore.profiler.profiling import Profiler
from mindspore.common.tensor import Tensor

from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.initializer import default_recurisive_init, load_resume_params
from src.logger import get_logger
from src.network_blocks import use_syc_bn
from src.util import get_param_groups, YOLOXCB, get_lr, load_backbone, EvalCallBack, DetectionEngine
from src.yolox import YOLOLossCell, TrainOneStepWithEMA, DetectionBaseBlock, DetectionTrainBlock, \
    DetectionTestBlock
from src.yolox_dataset import create_yolox_dataset

set_seed(42)


def set_default():
    """ set default """
    if config.enable_modelarts:
        config.data_root = os.path.join(config.data_dir, 'train2017')
        config.annFile = os.path.join(config.data_dir, 'annotations')
        outputs_dir = os.path.join(config.outputs_dir, config.ckpt_path)
    else:
        config.data_root = os.path.join(config.data_dir, 'train2017')
        config.annFile = os.path.join(config.data_dir, 'annotations/instances_train2017.json')
        outputs_dir = config.ckpt_path

    # logger
    config.outputs_dir = os.path.join(outputs_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)


def set_graph_kernel_context():
    if context.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_parallel_fusion "
                                               "--enable_trans_op_optimize "
                                               "--disable_cluster_ops=ReduceMax,Reshape "
                                               "--enable_expand_ops=Conv2D")


def network_init(cfg):
    """ Network init """
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, save_graphs=cfg.save_graphs, device_id=device_id,
                        save_graphs_path="ir_path")
    set_graph_kernel_context()

    profiler = None
    if cfg.need_profiler:
        profiling_dir = os.path.join(cfg.outputs_dir,
                                     datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        profiler = Profiler(output_path=profiling_dir, is_detail=True, is_show_op_path=True)

    # init distributed
    cfg.use_syc_bn = False
    if cfg.is_distributed:
        cfg.use_syc_bn = True
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=cfg.group_size)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    cfg.rank_save_ckpt_flag = 0
    if cfg.is_save_on_master:
        if cfg.rank == 0:
            cfg.rank_save_ckpt_flag = 1
    else:
        cfg.rank_save_ckpt_flag = 1

    # logger
    cfg.outputs_dir = os.path.join(cfg.ckpt_path,
                                   datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    cfg.logger = get_logger(cfg.outputs_dir, cfg.rank)
    cfg.logger.save_args(cfg)
    return profiler


def parallel_init(args):
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)


def modelarts_pre_process():
    '''modelarts pre process function.'''

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
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


def parser_init():
    parser = argparse.ArgumentParser(description='Yolox train.')
    parser.add_argument('--data_url', required=False, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=False, default=None, help='Location of training outputs.')
    parser.add_argument('--backbone', required=False, default="yolox_darknet53")
    parser.add_argument('--min_lr_ratio', required=False, default=0.05)
    parser.add_argument('--data_aug', required=False, default=True)
    return parser


def get_val_dataset():
    val_root = os.path.join(config.data_dir, 'val2017')
    ann_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    ds_test = create_yolox_dataset(val_root, ann_file, is_training=False, batch_size=config.per_batch_size,
                                   device_num=config.group_size,
                                   rank=config.rank)
    config.logger.info("Finish loading the val dataset!")
    return ds_test


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """ Launch Train process """
    parser = parser_init()
    args_opt, _ = parser.parse_known_args()
    set_default()
    if not config.data_aug:  # Train the last no data augment epochs
        config.use_l1 = True  # Add L1 loss
        config.max_epoch = config.no_aug_epochs
        config.lr_scheduler = "no_aug_lr"  # fix the min lr for last no data aug epochs
    if config.enable_modelarts:
        import moxing as mox
        local_data_url = os.path.join(config.data_path, str(config.rank))
        local_annFile = os.path.join(config.data_path, str(config.rank))
        mox.file.copy_parallel(config.data_root, local_data_url)
        config.data_root = local_data_url
        mox.file.copy_parallel(config.annFile, local_annFile)
        config.annFile = os.path.join(local_data_url, 'instances_train2017.json')
    profiler = network_init(config)
    parallel_init(config)
    if config.backbone == "yolox_darknet53":
        backbone = "yolofpn"
    else:
        backbone = "yolopafpn"
    base_network = DetectionBaseBlock(config, backbone=backbone)
    test_network = DetectionTestBlock(base_network)
    network = DetectionTrainBlock(base_network)  # train network
    if config.pretrained:
        network = load_backbone(network, config.pretrained, config)
    config.logger.info('Training backbone is: %s' % config.backbone)
    if config.use_syc_bn:
        config.logger.info("Using Synchronized batch norm layer...")
        use_syc_bn(network)
    default_recurisive_init(network)
    config.logger.info("Network weights have been initialized...")
    network = YOLOLossCell(network, config)
    config.logger.info('Finish getting network...')
    config.data_root = os.path.join(config.data_dir, 'train2017')
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_train2017.json')
    ds = create_yolox_dataset(image_dir=config.data_root,
                              anno_path=config.annFile,
                              batch_size=config.per_batch_size,
                              device_num=config.group_size,
                              rank=config.rank,
                              data_aug=config.data_aug)
    ds_test = get_val_dataset()
    config.logger.info('Finish loading training dataset! batch size:%s' % config.per_batch_size)
    config.steps_per_epoch = ds.get_dataset_size()
    config.logger.info('%s steps for one epoch.' % config.steps_per_epoch)
    if config.ckpt_interval <= 0:
        config.ckpt_interval = 1
    lr = get_lr(config)
    config.logger.info("Learning rate scheduler:%s, base_lr:%s, min lr ratio:%s" % (config.lr_scheduler, config.lr,
                                                                                    config.min_lr_ratio))
    opt = Momentum(params=get_param_groups(network, config.weight_decay),
                   learning_rate=Tensor(lr),
                   momentum=config.momentum,
                   use_nesterov=True)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 22)
    update_cell = loss_scale_manager.get_update_cell()
    network_ema = TrainOneStepWithEMA(network, opt, update_cell,
                                      ema=True, decay=0.9999, updates=0).set_train()
    if config.resume_yolox:
        load_resume_params(config, network_ema)
    if not config.data_aug:
        if os.path.isfile(config.yolox_no_aug_ckpt):  # Loading the resume checkpoint for the last no data aug epochs
            param_dict = load_checkpoint(config.yolox_no_aug_ckpt)
            if "learning_rate" in param_dict:
                param_dict.pop("learning_rate")
            if "global_step" in param_dict:
                param_dict.pop("global_step")
            load_param_into_net(network_ema, param_dict)
            config.logger.info("Finish load the resume checkpoint, begin to train the last...")
        else:
            raise FileNotFoundError('{} not exist or not a pre-trained file'.format(config.yolox_no_aug_ckpt))
    config.logger.info("Add ema model")
    model = Model(network_ema, amp_level="O0")
    cb = [SummaryCollector(summary_dir="./summary_dir", collect_freq=1)]
    save_ckpt_path = None
    if config.rank_save_ckpt_flag:
        ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch * config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        cb.append(ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank)))
    cb.append(YOLOXCB(config.logger, config.steps_per_epoch, lr=lr, save_ckpt_path=save_ckpt_path,
                      is_modelart=config.enable_modelarts,
                      per_print_times=config.log_interval, train_url=args_opt.train_url))
    cb.append(EvalCallBack(ds_test, test_network, DetectionEngine(config), config.logger, interval=513))
    if config.need_profiler:
        model.train(3, ds, callbacks=cb, dataset_sink_mode=True, sink_size=config.log_interval)
        profiler.analyse()
    else:
        config.logger.info("Epoch number:%s" % config.max_epoch)
        config.logger.info("All steps number:%s" % (config.max_epoch * config.steps_per_epoch))
        config.logger.info("==================Start Training=========================")
        model.train(config.max_epoch * config.steps_per_epoch // config.log_interval, ds, callbacks=cb,
                    dataset_sink_mode=True, sink_size=config.log_interval)
    config.logger.info("==================Training END======================")


if __name__ == "__main__":
    run_train()
