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
""" utils """
import os
import sys
import re
import time
import math
import json
import copy
from datetime import datetime
from collections import Counter
from typing import Optional, List, Union
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint, Parameter, Tensor
from mindspore.train.callback import Callback
from mindspore import ops
try:
    from third_party.fast_coco_eval_api import Fast_COCOeval as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from src.transform import xyxy2xywh
from model_utils.config import Config


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma ** milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def multi_step_lr(lr, milestones, steps_per_epoch, max_epoch, gamma=0.1):
    return warmup_step_lr(lr, milestones, steps_per_epoch, 0, max_epoch, gamma=gamma)


def step_lr(lr, epoch_size, steps_per_epoch, max_epoch, gamma=0.1):
    lr_epochs = []
    for i in range(1, max_epoch):
        if i % epoch_size == 0:
            lr_epochs.append(i)
    return multi_step_lr(lr, lr_epochs, steps_per_epoch, max_epoch, gamma=gamma)


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def yolox_warm_cos_lr(
        lr,
        steps_per_epoch,
        warmup_epochs,
        max_epoch,
        start_epoch,
        no_aug_epochs,
        warmup_lr_start=0,
        min_lr_ratio=0.05
):
    """Cosine learning rate with warm up."""
    base_lr = lr
    min_lr = lr * min_lr_ratio
    total_iters = int(max_epoch * steps_per_epoch)
    warmup_total_iters = int(warmup_epochs * steps_per_epoch)
    no_aug_iter = no_aug_epochs * steps_per_epoch
    lr_each_step = []
    for i in range(total_iters):
        if i < warmup_total_iters:
            lr = (base_lr - warmup_lr_start) * pow(
                (i + 1) / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif i >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(
                math.pi * (i - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
        lr_each_step.append(lr)
    lr_each_step = lr_each_step[start_epoch * steps_per_epoch:]
    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_v2(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate V2."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    last_lr = 0
    last_epoch_v1 = 0

    t_max_v2 = int(max_epoch * 1 / 3)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            if i < total_steps * 2 / 3:
                lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
                last_lr = lr
                last_epoch_v1 = last_epoch
            else:
                base_lr = last_lr
                last_epoch = last_epoch - last_epoch_v1
                lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max_v2)) / 2

        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_sample(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Warmup cosine annealing learning rate."""
    start_sample_epoch = 60
    step_sample = 2
    tobe_sampled_epoch = 60
    end_sampled_epoch = start_sample_epoch + step_sample * tobe_sampled_epoch
    max_sampled_epoch = max_epoch + tobe_sampled_epoch
    t_max = max_sampled_epoch

    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    total_sampled_steps = int(max_sampled_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []

    for i in range(total_sampled_steps):
        last_epoch = i // steps_per_epoch
        if last_epoch in range(start_sample_epoch, end_sampled_epoch, step_sample):
            continue
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    assert total_steps == len(lr_each_step)
    return np.array(lr_each_step).astype(np.float32)


def get_lr(args):
    """generate learning rate."""
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(args.lr,
                                        args.steps_per_epoch,
                                        args.warmup_epochs,
                                        args.max_epoch,
                                        args.t_max,
                                        args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_V2':
        lr = warmup_cosine_annealing_lr_v2(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.t_max,
                                           args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_sample':
        lr = warmup_cosine_annealing_lr_sample(args.lr,
                                               args.steps_per_epoch,
                                               args.warmup_epochs,
                                               args.max_epoch,
                                               args.t_max,
                                               args.eta_min)
    elif args.lr_scheduler == 'yolox_warm_cos_lr':
        lr = yolox_warm_cos_lr(lr=args.lr,
                               steps_per_epoch=args.steps_per_epoch,
                               warmup_epochs=args.warmup_epochs,
                               max_epoch=args.max_epoch,
                               start_epoch=args.start_epoch,
                               no_aug_epochs=args.no_aug_epochs,
                               min_lr_ratio=args.min_lr_ratio)
    else:
        raise NotImplementedError(args.lr_scheduler)
    return lr


def get_param_groups(network, weight_decay, use_group_params=True):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)
    if use_group_params:
        return [{'params': no_decay_params, 'weight_decay': 0.0},
                {'params': decay_params, 'weight_decay': weight_decay}]
    return network.trainable_params()


def load_weights(net, ckpt_path, pretrained=False):
    """Load darknet53 backbone checkpoint."""
    checkpoint_param = load_checkpoint(ckpt_path)
    ema_param_dict = dict()
    param_dict = dict()

    for param in checkpoint_param:
        if param.startswith("ema.network"):
            new_name = param.split("ema.")[1]
            ema_data = checkpoint_param[param]
            ema_data.name = new_name
            if re.search('cls_preds|reg_preds|obj_preds', new_name) and pretrained:
                continue
            ema_param_dict[new_name] = ema_data
        elif param.startswith('network.'):
            if re.search('cls_preds|reg_preds|obj_preds', param) and pretrained:
                continue
            param_dict[param] = checkpoint_param[param]

    if ema_param_dict:
        load_param_into_net(net, ema_param_dict)
    else:
        load_param_into_net(net, param_dict)
    return net


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        print("loss update----------------------------------------------------------------------")
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def keep_loss_fp32(network):
    """Keep loss of network with float32"""
    from src.yolox import YOLOLossCell
    for _, cell in network.cells_and_names():
        if isinstance(cell, (YOLOLossCell,)):
            cell.to_float(mstype.float32)


class ResumeCallback(Callback):
    def __init__(self, start_epoch=0):
        super(ResumeCallback, self).__init__()
        self.start_epoch = start_epoch

    def epoch_begin(self, run_context):
        run_context.original_args().cur_epoch_num += self.start_epoch


class NoAugCallBack(Callback):
    def __init__(self, use_l1=True):
        super(NoAugCallBack, self).__init__()
        self.use_l1 = use_l1

    def begin(self, run_context):
        run_context.original_args().network.network.use_l1 = self.use_l1


class YOLOXCB(Callback):
    """
    YOLOX Callback.
    """

    def __init__(self, config, lr, is_modelart=False, per_print_times=1, train_url=None):
        super(YOLOXCB, self).__init__()
        self.train_url = train_url
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.lr = lr
        self.is_modelarts = config.is_modelart
        self.step_per_epoch = config.steps_per_epoch
        self.logger = config.logger
        self.save_ckpt_path = config.save_ckpt_dir
        self.max_epoch = config.max_epoch
        self.is_modelarts = is_modelart
        self.current_step = 0
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.average_loss = []

    def epoch_begin(self, run_context):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        epoch_time = time.time() - self.epoch_start_time
        avg_step_time = epoch_time * 1000 / self.step_per_epoch
        loss = "loss: %.4f, overflow: %s, scale: %s" % (float(loss[0].asnumpy()),
                                                        bool(loss[1].asnumpy()),
                                                        int(loss[2].asnumpy()))
        self.logger.info("epoch: [%s/%s] epoch time: %.2fs %s avg step time: %.2fms" % (
            cur_epoch, self.max_epoch, epoch_time, loss, avg_step_time))

        if self.current_step % (self.step_per_epoch * 1) == 0:
            if self.is_modelarts:
                import moxing as mox
                if self.save_ckpt_path and self.train_url:
                    mox.file.copy_parallel(src_url=self.save_ckpt_path, dst_url=self.train_url)
                    cur_epoch = self.current_step // self.step_per_epoch
                    self.logger.info(
                        "[epoch {}]copy ckpt from{} to {}".format(self.save_ckpt_path, cur_epoch, self.train_url))

    def step_begin(self, run_context):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_end(self, run_context):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        data_sink_mode = cb_params.dataset_sink_mode
        cur_epoch_step = (self.current_step + 1) % self.step_per_epoch
        if cur_epoch_step % self._per_print_times == 0 and cur_epoch_step != 0 and not data_sink_mode:
            cur_epoch = cb_params.cur_epoch_num
            avg_step_time = (time.time() - self.step_start_time) * 1000 / self._per_print_times
            loss = cb_params.net_outputs
            loss = "loss: %.4f, overflow: %s, scale: %s" % (float(loss[0].asnumpy()),
                                                            bool(loss[1].asnumpy()),
                                                            int(loss[2].asnumpy()))
            self.logger.info("epoch: [%s/%s] step: [%s/%s], lr: %.6f, %s, avg step time: %.2fms" % (
                cur_epoch, self.max_epoch, cur_epoch_step, self.step_per_epoch, self.lr[self.current_step],
                loss, avg_step_time))
            self.step_start_time = time.time()
        self.current_step += 1

    def end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """


class EvalWrapper:
    def __init__(self, config: Config, dataset, network, detection_engine, save_prefix: Optional[str] = None) -> None:
        self.logger = config.logger
        self.dataset = dataset

        self.network = network
        self.detection_engine = detection_engine

        self.eval_parallel: bool = config.eval_parallel
        self.rank_id: int = config.rank
        self.data_list = []
        self.img_ids = []
        self.device_num: int = config.group_size
        self.save_prefix = config.outputs_dir if save_prefix is None else save_prefix
        self.file_path = ''
        self.dir_path = ''
        if config.eval_parallel:
            self.reduce = AllReduce()

    def synchronize(self):
        sync = Tensor(np.array([1]).astype(np.int32))
        sync = self.reduce(sync)    # For synchronization
        sync = sync.asnumpy()[0]
        if sync != self.device_num:
            raise ValueError(
                f"Sync value {sync} is not equal to number of device {self.device_num}. "
                f"There might be wrong with devices."
            )

    def inference_step(self, idx, data):
        image = data['image']
        img_shape = data['image_shape']
        img_id = data['img_id']
        prediction = self.network(image)
        prediction = prediction.asnumpy()
        img_shape = img_shape.asnumpy()
        img_id = img_id.asnumpy()
        if self.eval_parallel:
            mask = np.isin(img_id.squeeze(), self.img_ids, invert=True)
            prediction, img_shape, img_id = prediction[mask], img_shape[mask], img_id[mask]
            self.img_ids.extend(img_id.tolist())
        if prediction.shape[0] > 0:
            data = self.detection_engine.detection(prediction, img_shape, img_id)
            self.data_list.extend(data)
        self.logger.info(f'Detection...{idx} / {self.dataset.get_dataset_size()}')

    def inference(self, cur_epoch=0, cur_step=0):
        self.network.set_train(False)
        dataset_size = self.dataset.get_dataset_size()
        self.logger.info('Start inference...')
        self.logger.info(f"eval dataset size, {dataset_size}")
        self.img_ids.clear()
        for idx, data in enumerate(self.dataset.create_dict_iterator(num_epochs=1), start=1):
            self.inference_step(idx, data)
        self.save_prediction(cur_epoch, cur_step)
        result_file_path = self.file_path
        if self.eval_parallel:
            # Only support multiple devices on single machine
            self.synchronize()
            file_path = os.listdir(self.dir_path)
            result_file_path = [os.path.join(self.dir_path, path) for path in file_path]
        eval_result, results = self.detection_engine.get_eval_result(result_file_path)
        if eval_result is not None and results is not None:
            eval_print_str = '\n=============coco eval result=========\n' + eval_result
            return eval_print_str, results
        return None, 0

    def save_prediction(self, cur_epoch=0, cur_step=0):
        self.logger.info("Save bbox prediction result.")
        if self.eval_parallel:
            rank_id = self.rank_id
            self.dir_path = os.path.join(self.save_prefix, f"eval_epoch{cur_epoch}-step{cur_step}")
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path, exist_ok=True)
            file_name = f"epoch{cur_epoch}-step{cur_step}-rank{rank_id}.json"
            self.file_path = os.path.join(self.dir_path, file_name)
        else:
            t = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
            self.file_path = self.save_prefix + '/predict' + t + '.json'
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data_list, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What():{}".format(str(e)))
        else:
            self.data_list.clear()


class EvalCallback(Callback):
    def __init__(self, config: Config, eval_wrapper: EvalWrapper):
        self.logger = config.logger
        self.start_epoch = config.start_epoch
        self.max_epoch = config.max_epoch
        self.use_ema = config.use_ema
        self.train_aug_epochs = config.train_aug_epochs
        self.save_ckpt_path = config.save_ckpt_dir
        self.rank = config.rank
        self.resume_yolox = config.resume_yolox
        self.interval = config.eval_interval
        self.best_result = 0
        self.best_epoch = 0
        self.eval_wrapper: EvalWrapper = eval_wrapper

    def load_ema_parameter(self, network):
        param_dict = {}
        for name, param in network.parameters_and_names():
            if "ema." in name:
                new_name = name.split('ema.')[-1]
                param_new = param.clone()
                param_new.name = new_name
                param_dict[new_name] = param_new
        load_param_into_net(self.eval_wrapper.network, param_dict)

    def load_network_parameter(self, network):
        param_dict = {}
        for name, param in network.parameters_and_names():
            if name.startswith("network."):
                param_new = param.clone()
                param_dict[name] = param_new
        load_param_into_net(self.eval_wrapper.network, param_dict)

    def begin(self, run_context):
        best_ckpt_path = os.path.join(self.save_ckpt_path, 'best.ckpt')
        if os.path.exists(best_ckpt_path) and self.resume_yolox:
            param_dict = load_checkpoint(best_ckpt_path)
            self.best_result = param_dict['best_result'].asnumpy().item()
            self.best_epoch = param_dict['best_epoch'].asnumpy().item()
            self.logger.info('cur best result %s at epoch %s' % (self.best_result, self.best_epoch))

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.interval == 0 or cur_epoch == self.start_epoch + self.train_aug_epochs:
            if self.use_ema:
                self.load_ema_parameter(cb_param.train_network)
            else:
                self.load_network_parameter(cb_param.train_network)
            eval_print_str, results = self.inference(run_context)
            if results >= self.best_result:
                self.best_result = results
                self.best_epoch = cur_epoch
                if self.rank == 0:
                    self.save_best_checkpoint(cb_param.train_network)
                self.logger.info("Best result %s at %s epoch" % (self.best_result, self.best_epoch))
            self.logger.info(eval_print_str)
            self.logger.info('Ending inference...')

    def end(self, run_context):
        self.logger.info("Best result %s at %s epoch" % (self.best_result, self.best_epoch))

    def inference(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = cb_params.cur_step_num
        return self.eval_wrapper.inference(cur_epoch, cur_step)

    def save_best_checkpoint(self, net):
        param_list = [{'name': 'best_result', 'data': Parameter(self.best_result)},
                      {'name': 'best_epoch', 'data': Parameter(self.best_epoch)}]
        prefix = "network."
        for name, param in net.parameters_and_names():
            # Remove duplicate prefix 'network.'
            param_list.append({'name': name[len(prefix):], 'data': param})
        save_checkpoint(param_list, os.path.join(self.save_ckpt_path, 'best.ckpt'))


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class COCOData:
    """Class to save COCO related variables."""

    def __init__(self, ann_path: str) -> None:
        self.ann_path: str = ann_path
        self.coco: COCO = COCO(self.ann_path)
        self.img_ids: List[int] = list(sorted(self.coco.imgs.keys()))
        self.cat_ids: List[int] = self.coco.getCatIds()


class DetectionEngine:
    """ Detection engine """

    def __init__(self, config: Config):
        self.config = config
        self.input_size = self.config.input_size
        self.strides = self.config.fpn_strides  # [8, 16, 32]

        self.expanded_strides = None
        self.grids = None

        self.num_classes = config.num_classes

        self.conf_thre = config.conf_thre
        self.nms_thre = config.nms_thre
        self.coco_data = COCOData(os.path.join(config.data_dir, 'annotations/instances_val2017.json'))

    def detection(self, outputs, img_shape, img_ids):
        # post process nms
        outputs = self.postprocess(outputs, self.num_classes, self.conf_thre, self.nms_thre)
        return self.convert_to_coco_format(outputs, img_shape, img_ids)

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        """ nms """
        box_corner = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            if not image_pred.shape[0]:
                continue
            # Get score and class with highest confidence
            class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=-1)  # (8400)
            class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=-1)  # (8400)
            conf_mask = (image_pred[:, 4] * class_conf >= conf_thre).squeeze()  # (8400)
            class_conf = np.expand_dims(class_conf, axis=-1)  # (8400, 1)
            class_pred = np.expand_dims(class_pred, axis=-1).astype(np.float16)  # (8400, 1)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
            detections = detections[conf_mask]
            if not detections.shape[0]:
                continue
            if class_agnostic:
                nms_out_index = self._nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
            else:
                nms_out_index = self._batch_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                detections[:, 6], nms_thre)
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = np.concatenate((output[i], detections))
        return output

    @staticmethod
    def _nms(xyxys, scores, threshold):
        """Calculate NMS"""
        x1 = xyxys[:, 0]
        y1 = xyxys[:, 1]
        x2 = xyxys[:, 2]
        y2 = xyxys[:, 3]
        scores = scores
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h

            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)
            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _batch_nms(self, xyxys, scores, idxs, threshold, use_offset=True):
        """Calculate Nms based on class info,Each index value correspond to a category,
        and NMS will not be applied between elements of different categories."""
        if use_offset:
            max_coordinate = xyxys.max()
            offsets = idxs * (max_coordinate + np.array([1]))
            boxes_for_nms = xyxys + offsets[:, None]
            keep = self._nms(boxes_for_nms, scores, threshold)
            return keep
        keep_mask = np.zeros_like(scores, dtype=np.bool_)
        for class_id in np.unique(idxs):
            curr_indices = np.where(idxs == class_id)[0]
            curr_keep_indices = self._nms(xyxys[curr_indices], scores[curr_indices], threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = np.where(keep_mask)[0]
        return keep_indices[np.argsort(-scores[keep_indices])]

    def convert_to_coco_format(self, outputs, img_shapes, ids):
        """ convert to coco format """
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
                outputs, img_shapes[:, 0], img_shapes[:, 1], ids
        ):
            if output is None:
                continue
            bboxes = output[:, 0:4]
            scale = min(
                self.input_size[0] / float(img_h), self.input_size[1] / float(img_w)
            )

            bboxes = bboxes / scale
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_h)
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.coco_data.cat_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "score": scores[ind].item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def get_dt_list(self, file_path: List[str]):
        dt_list = []
        dt_ids_set = set([])
        self.config.logger.info(f"Total {len(file_path)} json files")
        self.config.logger.info(f"File list: {file_path}")

        for path in file_path:
            ann_list = []
            try:
                with open(path, 'r') as f:
                    ann_list = json.load(f)
            except json.decoder.JSONDecodeError:
                pass    # json file is empty
            else:
                ann_ids = set(ann['image_id'] for ann in ann_list)
                diff_ids = ann_ids - dt_ids_set
                ann_list = [ann for ann in ann_list if ann['image_id'] in diff_ids]
                dt_ids_set = dt_ids_set | diff_ids
                dt_list.extend(ann_list)
        return dt_list

    def get_coco_from_dt_list(self, dt_list) -> COCO:
        cocoDt = COCO()
        cocoDt.dataset = {}
        cocoDt.dataset['images'] = [img for img in self.coco_data.coco.dataset['images']]
        cocoDt.dataset['categories'] = copy.deepcopy(self.coco_data.coco.dataset['categories'])
        self.config.logger.info(f"Number of dt boxes: {len(dt_list)}")
        for idx, ann in enumerate(dt_list):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = idx + 1
            ann['iscrowd'] = 0
        cocoDt.dataset['annotations'] = dt_list
        cocoDt.createIndex()
        return cocoDt

    def compute_coco(self, cocoGt, cocoDt):
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        cocoEval.summarize()
        sys.stdout = stdout
        return rdct.content, cocoEval.stats[0]

    def get_eval_result(self, file_path: Union[str, List[str]]):
        """Get eval result"""
        if file_path is None:
            return None, None
        cocoGt = self.coco_data.coco
        if isinstance(file_path, str):
            cocoDt = cocoGt.loadRes(file_path)
        elif isinstance(file_path, list):
            dt_list = self.get_dt_list(file_path)
            cocoDt = self.get_coco_from_dt_list(dt_list)
        return self.compute_coco(cocoGt, cocoDt)


def get_specified():
    res = [
        'network.backbone.backbone.stem.0.conv.weight',
        'network.backbone.backbone.dark2.0.conv.weight',
        'network.backbone.backbone.dark3.0.conv.weight',
        'network.backbone.backbone.dark3.8.layer2.conv.weight',
        'network.backbone.backbone.dark4.0.conv.weight',
        'network.backbone.backbone.dark4.8.layer2.conv.weight',
        'network.backbone.backbone.dark5.0.conv.weight',
        'network.backbone.backbone.dark5.7.conv1.conv.weight',
        'network.backbone.backbone.dark5.7.conv2.conv.weight',
        'network.backbone.backbone.dark5.9.conv.weight',
        'network.head_l.cls_preds.weight',
        'network.head_m.cls_preds.weight',
        'network.head_s.cls_preds.weight',
        'network.head_l.reg_preds.weight',
        'network.head_m.reg_preds.weight',
        'network.head_s.reg_preds.weight',
        'network.head_l.obj_preds.weight',
        'network.head_m.obj_preds.weight',
        'network.head_s.obj_preds.weight',
    ]

    return res


class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce()

    def construct(self, x):
        return self.all_reduce(x)
