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
import time
import math
import json
import stat
from datetime import datetime
from collections import Counter
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint, Tensor, Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.train.callback import Callback
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.transform import xyxy2xywh


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


def yolox_no_aug_lr(base_lr, steps_per_epoch, max_epoch, min_lr_ratio=0.05):
    total_iters = int(max_epoch * steps_per_epoch)
    lr = base_lr * min_lr_ratio
    lr_each_step = []
    for _ in range(total_iters):
        lr_each_step.append(lr)
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
                               max_epoch=args.total_epoch,
                               no_aug_epochs=args.no_aug_epochs,
                               min_lr_ratio=args.min_lr_ratio)
    elif args.lr_scheduler == 'no_aug_lr':
        lr = yolox_no_aug_lr(
            args.lr,
            args.steps_per_epoch,
            args.max_epoch,
            min_lr_ratio=args.min_lr_ratio
        )
    else:
        raise NotImplementedError(args.lr_scheduler)
    return lr


def get_param_groups(network, weight_decay):
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

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params, 'weight_decay': weight_decay}]


def load_backbone(net, ckpt_path, args):
    """Load darknet53 backbone checkpoint."""
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in param_dict:
            pass
        else:
            param_not_load.append(param.name)
    args.logger.info("not loading param is :", len(param_not_load))
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


class EMACallBack(Callback):

    def __init__(self, network, steps_per_epoch, cur_steps=0):
        self.steps_per_epoch = steps_per_epoch
        self.cur_steps = cur_steps
        self.network = network

    def epoch_begin(self, run_context):
        if self.network.ema:
            if not isinstance(self.network.ema_moving_weight, list):
                tmp_moving = []
                for weight in self.network.ema_moving_weight:
                    tmp_moving.append(weight.asnumpy())
                self.network.ema_moving_weight = tmp_moving

    def step_end(self, run_context):
        if self.network.ema:
            self.network.moving_parameter_update()
            self.cur_steps += 1

            if self.cur_steps % self.steps_per_epoch == 0:
                if isinstance(self.network.ema_moving_weight, list):
                    tmp_moving = []
                    moving_name = []
                    idx = 0
                    for key in self.network.moving_name:
                        moving_name.append(key)

                    for weight in self.network.ema_moving_weight:
                        param = Parameter(Tensor(weight), name=moving_name[idx])
                        tmp_moving.append(param)
                        idx += 1
                    self.network.ema_moving_weight = ParameterTuple(tmp_moving)


class YOLOXCB(Callback):
    """
    YOLOX Callback.
    """

    def __init__(self, logger, step_per_epoch, lr, save_ckpt_path, is_modelart=False, per_print_times=1,
                 train_url=None):
        super(YOLOXCB, self).__init__()
        self.train_url = train_url
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.lr = lr
        self.is_modelarts = is_modelart
        self.step_per_epoch = step_per_epoch
        self.current_step = 0
        self.save_ckpt_path = save_ckpt_path
        self.iter_time = time.time()
        self.epoch_start_time = time.time()
        self.average_loss = []
        self.logger = logger

    def epoch_begin(self, run_context):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()
        self.iter_time = time.time()

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        loss = "loss: %.4f, overflow: %s, scale: %s" % (float(loss[0].asnumpy()),
                                                        bool(loss[1].asnumpy()),
                                                        int(loss[2].asnumpy()))
        self.logger.info(
            "epoch: %s epoch time %.2fs %s" % (cur_epoch, time.time() - self.epoch_start_time, loss))

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

        cur_epoch_step = (self.current_step + 1) % self.step_per_epoch
        if cur_epoch_step % self._per_print_times == 0 and cur_epoch_step != 0:
            cb_params = run_context.original_args()
            cur_epoch = cb_params.cur_epoch_num
            loss = cb_params.net_outputs
            loss = "loss: %.4f, overflow: %s, scale: %s" % (float(loss[0].asnumpy()),
                                                            bool(loss[1].asnumpy()),
                                                            int(loss[2].asnumpy()))
            self.logger.info("epoch: %s step: [%s/%s], %s, lr: %.6f, avg step time: %.2f ms" % (
                cur_epoch, cur_epoch_step, self.step_per_epoch, loss, self.lr[self.current_step],
                (time.time() - self.iter_time) * 1000 / self._per_print_times))
            self.iter_time = time.time()
        self.current_step += 1

    def end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """


class EvalCallBack(Callback):
    def __init__(self, dataset, test_net, train_net, detection, config, start_epoch=0, interval=1):
        self.dataset = dataset
        self.network = train_net
        self.test_network = test_net
        self.detection = detection
        self.logger = config.logger
        self.start_epoch = start_epoch
        self.interval = interval
        self.max_epoch = config.max_epoch
        self.best_result = 0
        self.best_epoch = 0
        self.rank = config.rank

    def load_ema_parameter(self):
        param_dict = {}
        for name, param in self.network.parameters_and_names():
            if name.startswith("ema."):
                new_name = name.split('ema.')[-1]
                param_new = param.clone()
                param_new.name = new_name
                param_dict[new_name] = param_new
        load_param_into_net(self.test_network, param_dict)

    def load_network_parameter(self):
        param_dict = {}
        for name, param in self.network.parameters_and_names():
            if name.startswith("network."):
                param_new = param.clone()
                param_dict[name] = param_new
        load_param_into_net(self.test_network, param_dict)

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch >= self.start_epoch:
            if (cur_epoch - self.start_epoch) % self.interval == 0 or cur_epoch == self.max_epoch:
                self.load_network_parameter()
                self.test_network.set_train(False)
                eval_print_str, results = self.inference()
                if results >= self.best_result:
                    self.best_result = results
                    self.best_epoch = cur_epoch
                    if os.path.exists('best.ckpt'):
                        self.remove_ckpoint_file('best.ckpt')
                    save_checkpoint(cb_param.train_network, 'best.ckpt')
                    self.logger.info("Best result %s at %s epoch" % (self.best_result, self.best_epoch))
                self.logger.info(eval_print_str)
                self.logger.info('Ending inference...')

    def end(self, run_context):
        self.logger.info("Best result %s at %s epoch" % (self.best_result, self.best_epoch))

    def inference(self):
        self.logger.info('Start inference...')
        self.logger.info("eval dataset size, %s" % self.dataset.get_dataset_size())
        counts = 0
        for data in self.dataset.create_dict_iterator(num_epochs=1):
            image = data['image']
            img_info = data['image_shape']
            img_id = data['img_id']
            prediction = self.test_network(image)
            prediction = prediction.asnumpy()
            img_shape = img_info.asnumpy()
            img_id = img_id.asnumpy()
            counts = counts + 1
            self.detection.detection(prediction, img_shape, img_id)
            self.logger.info('Calculating mAP...%s' % counts)

        self.logger.info('Calculating mAP...%s' % counts)
        result_file_path = self.detection.evaluate_prediction()
        self.logger.info('result file path: %s', result_file_path)
        eval_result, results = self.detection.get_eval_result()
        if eval_result is not None and results is not None:
            eval_print_str = '\n=============coco eval result=========\n' + eval_result
            return eval_print_str, results
        return None, 0

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            self.logger.info("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            self.logger.info("ValueError, failed to remove the older ckpt file %s.", file_name)


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class DetectionEngine:
    """ Detection engine """

    def __init__(self, config):
        self.config = config
        self.input_size = self.config.input_size
        self.strides = self.config.fpn_strides  # [8, 16, 32]

        self.expanded_strides = None
        self.grids = None

        self.num_classes = config.num_classes

        self.conf_thre = config.conf_thre
        self.nms_thre = config.nms_thre
        self.annFile = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
        self._coco = COCO(self.annFile)
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.coco_catIds = self._coco.getCatIds()
        self.save_prefix = config.outputs_dir
        self.file_path = ''

        self.data_list = []

    def detection(self, outputs, img_shape, img_ids):
        # post process nms
        outputs = self.postprocess(outputs, self.num_classes, self.conf_thre, self.nms_thre)
        self.data_list.extend(self.convert_to_coco_format(outputs, info_imgs=img_shape, ids=img_ids))

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

    def _nms(self, xyxys, scores, threshold):
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

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        """ convert to coco format """
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
                outputs, info_imgs[:, 0], info_imgs[:, 1], ids
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
                label = self.coco_catIds[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "score": scores[ind].item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self):
        """ generate prediction coco json file """
        print('Evaluate in main process...')
        # write result to coco json format

        t = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.data_list, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What():{}".format(str(e)))
        else:
            f.close()
            if not self.data_list:
                self.file_path = ''
                return self.file_path

            self.data_list.clear()
            return self.file_path

    def get_eval_result(self):
        """Get eval result"""
        if not self.file_path:
            return None, None

        cocoGt = self._coco
        cocoDt = cocoGt.loadRes(self.file_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        cocoEval.summarize()
        sys.stdout = stdout
        return rdct.content, cocoEval.stats[0]
