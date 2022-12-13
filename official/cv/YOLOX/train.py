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
import shutil
import time
import datetime
import argparse

import mindspore
from mindspore import DynamicLossScaleManager
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.profiler.profiling import Profiler
from mindspore.common.tensor import Tensor

from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.initializer import default_recurisive_init
from src.logger import get_logger
from src.network_blocks import use_syc_bn
from src.util import get_specified, get_param_groups, YOLOXCB, get_lr, load_weights, EvalCallback, DetectionEngine, \
    ResumeCallback, EvalWrapper, NoAugCallBack
from src.yolox import YOLOLossCell, TrainOneStepWithEMA, DetectionBlock
from src.yolox_dataset import create_yolox_dataset

set_seed(888)


def set_default():
    """ set default """
    if config.enable_modelarts:
        config.data_root = os.path.join(config.data_dir, 'coco2017/train2017')
        config.annFile = os.path.join(config.data_dir, 'coco2017/annotations')
        outputs_dir = os.path.join(config.outputs_dir, config.ckpt_path)
    else:
        config.data_root = os.path.join(config.data_dir, 'train2017')
        config.annFile = os.path.join(config.data_dir, 'annotations/instances_train2017.json')
        outputs_dir = os.getcwd()
        if config.resume_yolox:
            base_dir = os.path.abspath(config.resume_yolox)
            config.save_ckpt_dir = os.path.dirname(base_dir)
        else:
            config.save_ckpt_dir = os.path.join(config.ckpt_dir, config.backbone)
    if not os.path.exists(config.save_ckpt_dir):
        os.makedirs(config.save_ckpt_dir)


    # logger
    config.outputs_dir = os.path.join(outputs_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.max_epoch = config.aug_epochs + config.no_aug_epochs
    config.train_aug_epochs = config.aug_epochs
    config.train_no_aug_epochs = config.no_aug_epochs
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)


def set_graph_kernel_context():
    if context.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_parallel_fusion "
                                               "--enable_trans_op_optimize "
                                               "--disable_cluster_ops=ReduceMax,Reshape "
                                               "--enable_expand_ops=Conv2D")


def network_init():
    """ Network init """
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=config.save_graphs, device_id=device_id,
                        save_graphs_path="ir_path", max_call_depth=2000)
    set_graph_kernel_context()

    profiler = None
    if config.need_profiler:
        profiling_dir = os.path.join(config.outputs_dir,
                                     datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        profiler = Profiler(output_path=profiling_dir, is_detail=True, is_show_op_path=True)

    # init distributed
    if not config.use_syc_bn:
        config.use_syc_bn = False
    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=config.group_size)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    if config.is_save_on_master:
        config.rank_save_ckpt_flag = 0
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.outputs_dir,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)
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

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_dir)


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


def get_optimizer(cfg, network, lr):
    if cfg.opt == "SGD":
        from mindspore.nn import SGD
        # SGD grouping parameters is currently not supported
        params = get_param_groups(network, cfg.weight_decay, use_group_params=False)
        opt = SGD(params=params, learning_rate=Tensor(lr), momentum=config.momentum, weight_decay=config.weight_decay,
                  nesterov=True)
        cfg.logger.info("Use SGD Optimizer")
    else:
        from mindspore.nn import Momentum
        param_group = get_param_groups(network, cfg.weight_decay)
        opt = Momentum(params=param_group, learning_rate=Tensor(lr), momentum=cfg.momentum, use_nesterov=True)
        cfg.logger.info("Use Momentum Optimizer")
    return opt


class MySgdOptimizer(mindspore.nn.SGD):
    def __init__(self, *args, **kwargs):
        super(MySgdOptimizer, self).__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.param_length = 0
        self.param_mask = self.get_weight_params()

    def get_weight_params(self):
        param_mask = list()
        for i in range(len(self.parameters)):
            name = self.parameters[i].name
            self.param_length += 1
            if name.endswith('.weight'):
                param_mask.append(Tensor(0.0, mindspore.float32))
            else:
                param_mask.append(self.weight_decay)
        return param_mask

    def construct(self, gradients):
        new_grads = ()
        for i in range(self.param_length):
            grads_tmp = gradients[i] - self.param_mask[i] * self.parameters[i]
            new_grads += (grads_tmp,)
        return self._original_construct(new_grads)


def set_resume_config():
    if not os.path.isfile(config.resume_yolox):
        raise TypeError('resume_yolox should be checkpoint path')
    resume_param = load_checkpoint(config.resume_yolox,
                                   filter_prefix=['learning_rate', 'global_step', 'best_result', 'best_epoch'])
    ckpt_name = os.path.basename(config.resume_yolox)
    resume_epoch = ckpt_name.split('-')[1].split('_')[0]
    config.start_epoch = int(resume_epoch)
    config.train_aug_epochs = max(config.aug_epochs - config.start_epoch, 0)
    return resume_param, resume_epoch


def set_no_aug():
    config.use_l1 = True
    config.run_eval = True
    config.eval_interval = 1
    config.ckpt_interval = 1
    config.start_epoch = max(config.aug_epochs, config.start_epoch)
    config.train_no_aug_epochs = max(config.max_epoch - config.start_epoch, 0)


def set_modelarts_config():
    if config.enable_modelarts:
        import moxing as mox
        local_data_url = os.path.join(config.data_path, str(config.rank))
        local_annFile = os.path.join(config.data_path, str(config.rank))
        mox.file.copy_parallel(config.data_root, local_data_url)
        config.data_dir = os.path.join(config.data_path, 'coco2017')
        mox.file.copy_parallel(config.annFile, local_annFile)
        config.annFile = os.path.join(local_data_url, 'instances_train2017.json')


def set_callback(ds=None, backbone=None, lr=None, train_url=None):
    cb = []
    if config.rank_save_ckpt_flag:
        if config.use_summary:
            specified = {'collect_input_data': False, 'histogram_regular': '|'.join(get_specified())}
            cb.append(SummaryCollector(summary_dir="./summary_dir", collect_freq=10, collect_specified_data=specified))
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch * config.ckpt_interval,
                                       keep_checkpoint_max=config.ckpt_max_num)
        cb.append(
            ModelCheckpoint(config=ckpt_config, directory=config.save_ckpt_dir, prefix='{}'.format(config.backbone)))
    if config.resume_yolox or config.start_epoch:
        cb.append(ResumeCallback(config.start_epoch))

    if lr is not None:
        cb.append(YOLOXCB(config, lr=lr, is_modelart=config.enable_modelarts, per_print_times=config.log_interval,
                          train_url=train_url))
    if config.run_eval and backbone is not None and ds is not None:
        test_network = DetectionBlock(config, backbone=backbone)
        save_prefix = None
        if config.eval_parallel:
            save_prefix = config.eval_parallel_dir
            if os.path.exists(save_prefix):
                shutil.rmtree(save_prefix, ignore_errors=True)
        detection_engine = DetectionEngine(config)
        eval_wrapper = EvalWrapper(
            config=config,
            dataset=ds,
            network=test_network,
            detection_engine=detection_engine,
            save_prefix=save_prefix
        )
        cb.append(EvalCallback(config, eval_wrapper))
    return cb


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """ Launch Train process """
    parser = parser_init()
    args_opt, _ = parser.parse_known_args()
    set_default()
    set_modelarts_config()

    profiler = network_init()
    parallel_init(config)
    if config.backbone == "yolox_darknet53":
        backbone = "yolofpn"
    elif config.backbone == 'yolox_x':
        backbone = "yolopafpn"
    else:
        raise ValueError('backbone only support [yolox_darknet53, yolox_x]')
    base_network = DetectionBlock(config, backbone=backbone)

    # syc bn only support distributed training in graph mode
    if config.use_syc_bn and config.is_distributed and context.get_context('mode') == context.GRAPH_MODE:
        config.logger.info("Using Synchronized batch norm layer...")
        use_syc_bn(base_network)
    default_recurisive_init(base_network)
    config.logger.info("Network weights have been initialized...")
    if config.pretrained:
        base_network = load_weights(base_network, config.pretrained, pretrained=True)
        config.logger.info('pretrained is: %s' % config.pretrained)
    config.logger.info('Training backbone is: %s' % config.backbone)
    config.logger.info('Finish getting network...')

    if config.resume_yolox:
        resume_param, resume_epoch = set_resume_config()
        config.logger.info('resume train from epoch: %s data_aug: %s' % (resume_epoch, config.data_aug))
    network = YOLOLossCell(base_network, config)
    config.eval_parallel = config.run_eval and config.is_distributed and config.eval_parallel

    ds_augs = create_yolox_dataset(image_dir=config.data_root, anno_path=config.annFile,
                                   batch_size=config.per_batch_size, device_num=config.group_size, rank=config.rank,
                                   data_aug=True)
    ds_no_augs = create_yolox_dataset(image_dir=config.data_root, anno_path=config.annFile,
                                      batch_size=config.per_batch_size, device_num=config.group_size, rank=config.rank,
                                      data_aug=False)
    ds_test = get_val_dataset()
    config.logger.info('Finish loading training dataset! batch size:%s' % config.per_batch_size)
    config.steps_per_epoch = ds_augs.get_dataset_size()
    config.logger.info('%s steps for one epoch.' % config.steps_per_epoch)

    lr = get_lr(config)
    config.logger.info("Learning rate scheduler:%s, base_lr:%s, min lr ratio:%s" % (config.lr_scheduler, config.lr,
                                                                                    config.min_lr_ratio))
    opt = get_optimizer(config, network, lr)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 22)
    update_cell = loss_scale_manager.get_update_cell()
    network_ema = TrainOneStepWithEMA(network, opt, update_cell, ema=config.use_ema, decay=0.9998).set_train()
    if config.resume_yolox:
        load_param_into_net(network_ema, resume_param)
    config.logger.info("use ema model: %s" % config.use_ema)
    model = Model(network_ema)

    if config.need_profiler:
        model.train(3, ds, callbacks=cb, dataset_sink_mode=True, sink_size=config.log_interval)
        profiler.analyse()
    else:
        if config.train_aug_epochs:
            config.logger.info("Aug train epoch number:%s" % config.train_aug_epochs)
            config.logger.info("Aug train steps number:%s" % (config.train_aug_epochs * config.steps_per_epoch))
            config.logger.info("==================Start Training=========================")
            cb_augs = set_callback(ds=ds_test, backbone=backbone, lr=lr, train_url=args_opt.train_url)
            model.train(config.train_aug_epochs, ds_augs, callbacks=cb_augs, dataset_sink_mode=False, sink_size=-1)

        # train no augs use l1 loss
        set_no_aug()
        if config.train_no_aug_epochs:
            config.logger.info("No Aug train epoch number:%s" % config.train_no_aug_epochs)
            config.logger.info("No Aug train steps number:%s" % (config.train_no_aug_epochs * config.steps_per_epoch))
            config.logger.info("==================Start Training=========================")
            cb_no_augs = set_callback(ds=ds_test, backbone=backbone, lr=lr, train_url=args_opt.train_url)
            cb_no_augs.append(NoAugCallBack(config.use_l1))
            model.train(config.no_aug_epochs, ds_no_augs, callbacks=cb_no_augs, dataset_sink_mode=False, sink_size=-1)
    config.logger.info("==================Training END======================")


if __name__ == "__main__":
    run_train()
