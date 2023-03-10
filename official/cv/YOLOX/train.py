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
import copy
import time
import datetime

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
    ResumeCallback, EvalWrapper
from src.yolox import YOLOLossCell, TrainOneStepWithEMA, DetectionBlock
from src.yolox_dataset import create_yolox_dataset

set_seed(888)


def set_default(cfg):
    """ set default """
    if cfg.enable_modelarts:
        cfg.data_root = os.path.join(cfg.data_dir, 'coco2017/train2017')
        cfg.annFile = os.path.join(cfg.data_dir, 'coco2017/annotations')
        cfg.save_ckpt_dir = os.path.join(os.getcwd(), cfg.ckpt_dir)
    else:
        cfg.data_root = os.path.join(cfg.data_dir, 'train2017')
        cfg.annFile = os.path.join(cfg.data_dir, 'annotations/instances_train2017.json')
        if cfg.resume_yolox:
            base_dir = os.path.abspath(cfg.resume_yolox)
            cfg.save_ckpt_dir = os.path.dirname(base_dir)
        else:
            cfg.save_ckpt_dir = os.path.join(cfg.ckpt_dir, cfg.backbone)
    if not os.path.exists(cfg.save_ckpt_dir):
        os.makedirs(cfg.save_ckpt_dir)

    # logger
    rank = int(os.getenv('RANK_ID', '0'))
    cfg.log_dir = os.path.join(cfg.save_ckpt_dir, 'log', 'rank_%s' % rank)
    cfg.max_epoch = cfg.aug_epochs + cfg.no_aug_epochs
    cfg.train_aug_epochs = cfg.aug_epochs
    cfg.train_no_aug_epochs = cfg.no_aug_epochs
    cfg.logger = get_logger(cfg.log_dir, rank)
    return cfg


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
                        save_graphs_path="ir_path", max_call_depth=2000)
    set_graph_kernel_context()

    cfg.profiler = None
    if cfg.need_profiler:
        profiling_dir = os.path.join(cfg.log_dir,
                                     datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        cfg.profiler = Profiler(output_path=profiling_dir, is_detail=True, is_show_op_path=True)

    # init distributed
    if not cfg.use_syc_bn:
        cfg.use_syc_bn = False
    if cfg.is_distributed:
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=cfg.group_size)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    if cfg.is_save_on_master:
        cfg.rank_save_ckpt_flag = 0
        if cfg.rank == 0:
            cfg.rank_save_ckpt_flag = 1
    else:
        cfg.rank_save_ckpt_flag = 1
    return cfg


def parallel_init(args):
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)


def modelarts_pre_process(cfg):
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
        zip_file_1 = os.path.join(cfg.data_path, cfg.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(cfg.data_path)

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

    cfg.ckpt_path = os.path.join(cfg.output_path, cfg.ckpt_dir)
    return cfg


def get_val_dataset(cfg):
    val_root = os.path.join(cfg.data_dir, 'val2017')
    ann_file = os.path.join(cfg.data_dir, 'annotations/instances_val2017.json')
    ds_test = create_yolox_dataset(val_root, ann_file, is_training=False, batch_size=cfg.per_batch_size,
                                   device_num=cfg.group_size,
                                   rank=cfg.rank)
    cfg.logger.info("Finish loading the val dataset!")
    return ds_test


def get_optimizer(cfg, network, lr):
    if cfg.opt == "SGD":
        # nn.SGD grouping parameters with weight_decay is currently not supported
        params = get_param_groups(network, cfg.weight_decay, use_group_params=False)
        opt = MySgdOptimizer(params=params, learning_rate=Tensor(lr), momentum=cfg.momentum,
                             weight_decay=cfg.weight_decay, nesterov=True)
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


def set_resume_config(cfg):
    if not os.path.isfile(cfg.resume_yolox):
        raise TypeError('resume_yolox should be checkpoint path')
    resume_param = load_checkpoint(cfg.resume_yolox,
                                   choice_func=lambda x: not x.startswith(
                                       ('learning_rate', 'global_step', 'best_result',
                                        'best_epoch')))
    ckpt_name = os.path.basename(cfg.resume_yolox)
    resume_epoch = ckpt_name.split('-')[1].split('_')[0]
    cfg.start_epoch = int(resume_epoch)
    cfg.train_aug_epochs = max(cfg.aug_epochs - cfg.start_epoch, 0)
    return resume_param, resume_epoch, cfg


def set_no_aug(cfg):
    cfg.use_l1 = True
    cfg.run_eval = True
    cfg.eval_interval = 1
    cfg.ckpt_interval = 1
    cfg.start_epoch = max(cfg.aug_epochs, cfg.start_epoch)
    cfg.train_no_aug_epochs = max(cfg.max_epoch - cfg.start_epoch, 0)
    return cfg


def set_modelarts_config(cfg):
    if cfg.enable_modelarts:
        import moxing as mox
        local_data_url = os.path.join(cfg.data_path, str(cfg.rank))
        local_annFile = os.path.join(config.data_path, str(cfg.rank))
        mox.file.copy_parallel(cfg.data_root, local_data_url)
        cfg.data_dir = os.path.join(cfg.data_path, 'coco2017')
        mox.file.copy_parallel(cfg.annFile, local_annFile)
        cfg.annFile = os.path.join(local_data_url, 'instances_train2017.json')
        return cfg
    return cfg


def set_callback(cfg, ds=None, backbone='yolofpn'):
    cb = []
    if cfg.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.steps_per_epoch * cfg.ckpt_interval,
                                       keep_checkpoint_max=cfg.ckpt_max_num)
        cb.append(
            ModelCheckpoint(config=ckpt_config, directory=cfg.save_ckpt_dir, prefix='{}'.format(cfg.backbone)))
    if cfg.resume_yolox or cfg.start_epoch:
        cb.append(ResumeCallback(cfg.start_epoch))

    if cfg.run_eval and ds is not None:
        test_network = DetectionBlock(cfg, backbone=backbone)
        save_prefix = None
        if config.eval_parallel:
            save_prefix = cfg.eval_parallel_dir
            if os.path.exists(save_prefix):
                shutil.rmtree(save_prefix, ignore_errors=True)
        detection_engine = DetectionEngine(config)
        eval_wrapper = EvalWrapper(
            config=cfg,
            dataset=ds,
            network=test_network,
            detection_engine=detection_engine,
            save_prefix=save_prefix
        )
        cb.append(EvalCallback(config, eval_wrapper))
    return cb, cfg


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train(cfg):
    """ Launch Train process """
    cfg = set_default(cfg)
    cfg = set_modelarts_config(cfg)

    cfg = network_init(cfg)
    parallel_init(cfg)
    if cfg.backbone == "yolox_darknet53":
        backbone = "yolofpn"
    elif cfg.backbone == 'yolox_x':
        backbone = "yolopafpn"
    else:
        raise ValueError('backbone only support [yolox_darknet53, yolox_x]')
    base_network = DetectionBlock(cfg, backbone=backbone)

    # syc bn only support distributed training in graph mode
    if cfg.use_syc_bn and cfg.is_distributed and context.get_context('mode') == context.GRAPH_MODE:
        cfg.logger.info("Using Synchronized batch norm layer...")
        use_syc_bn(base_network)
    default_recurisive_init(base_network)
    cfg.logger.info("Network weights have been initialized...")
    if cfg.pretrained:
        base_network = load_weights(base_network, cfg.pretrained, pretrained=True)
        cfg.logger.info('pretrained is: %s' % cfg.pretrained)
    cfg.logger.info('Training backbone is: %s' % cfg.backbone)
    cfg.logger.info('Finish getting network...')

    if cfg.resume_yolox:
        resume_param, resume_epoch, cfg = set_resume_config(cfg)
        config.logger.info('resume train from epoch: %s' % resume_epoch)
    network = YOLOLossCell(base_network, cfg)
    cfg.eval_parallel = cfg.run_eval and cfg.is_distributed and cfg.eval_parallel

    ds_augs = create_yolox_dataset(image_dir=cfg.data_root, anno_path=cfg.annFile,
                                   batch_size=cfg.per_batch_size, device_num=cfg.group_size, rank=cfg.rank,
                                   data_aug=True)
    ds_no_augs = create_yolox_dataset(image_dir=cfg.data_root, anno_path=cfg.annFile,
                                      batch_size=cfg.per_batch_size, device_num=cfg.group_size, rank=cfg.rank,
                                      data_aug=False)
    ds_test = get_val_dataset(cfg)
    cfg.logger.info('Finish loading training dataset! batch size:%s' % cfg.per_batch_size)
    cfg.steps_per_epoch = ds_augs.get_dataset_size()
    cfg.logger.info('%s steps for one epoch.' % cfg.steps_per_epoch)

    lr = get_lr(cfg)
    cfg.logger.info(
        "Learning rate scheduler:%s, base_lr:%s, min lr ratio:%s" % (cfg.lr_scheduler, cfg.lr, cfg.min_lr_ratio))
    opt = get_optimizer(config, network, lr)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 22)
    update_cell = loss_scale_manager.get_update_cell()
    network_ema = TrainOneStepWithEMA(network, opt, update_cell, ema=cfg.use_ema, decay=0.9998).set_train()

    if cfg.resume_yolox:
        load_param_into_net(network_ema, resume_param)
    cfg.logger.info("use ema model: %s" % cfg.use_ema)
    model = Model(network_ema)
    cb_default = list()
    if cfg.rank_save_ckpt_flag:
        if cfg.use_summary:
            specified = {'collect_input_data': False, 'histogram_regular': '|'.join(get_specified())}
            cb_default.append(
                SummaryCollector(summary_dir="./summary_dir", collect_freq=10, collect_specified_data=specified))
    yolo_cb = YOLOXCB(cfg, lr=lr, is_modelart=cfg.enable_modelarts, per_print_times=cfg.log_interval,
                      train_url=cfg.train_url)
    if config.need_profiler:
        model.train(3, ds_augs, callbacks=cb_default, dataset_sink_mode=cfg.dataset_sink_mode,
                    sink_size=config.log_interval)
        cfg.profiler.analyse()
    else:
        if cfg.train_aug_epochs:
            cfg.logger.save_args(cfg)
            cfg.logger.info("Aug train epoch number:%s" % cfg.train_aug_epochs)
            cfg.logger.info("Aug train steps number:%s" % (cfg.train_aug_epochs * cfg.steps_per_epoch))
            cfg.logger.info("==================Start Training=========================")
            cb_augs, cfg = set_callback(cfg, ds=ds_test, backbone=backbone)
            cb_augs = cb_augs + cb_default + [yolo_cb]
            model.train(cfg.train_aug_epochs, ds_augs, callbacks=cb_augs, dataset_sink_mode=cfg.dataset_sink_mode,
                        sink_size=-1)

        # use l1 and run eval train no augs
        cfg = set_no_aug(cfg)
        if cfg.train_no_aug_epochs:
            cfg.logger.save_args(cfg)
            cfg.logger.info("No Aug train epoch number:%s" % cfg.train_no_aug_epochs)
            cfg.logger.info("No Aug train steps number:%s" % (cfg.train_no_aug_epochs * cfg.steps_per_epoch))
            cfg.logger.info("==================Start Training=========================")
            cb_no_augs, cfg = set_callback(cfg, ds=ds_test, backbone=backbone)
            cb_no_augs = cb_no_augs + cb_default + [yolo_cb]

            network_ema_no_augs = copy.copy(network_ema)
            network_ema_no_augs.network.use_l1 = cfg.use_l1
            model_no_augs = Model(network_ema_no_augs)

            model_no_augs.train(cfg.no_aug_epochs, ds_no_augs, callbacks=cb_no_augs,
                                dataset_sink_mode=cfg.dataset_sink_mode, sink_size=-1)
            cfg.logger.info("==================Training END======================")


if __name__ == "__main__":
    run_train(config)
