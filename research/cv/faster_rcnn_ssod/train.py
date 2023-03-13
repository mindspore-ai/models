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
"""Train"""
import os
import time
from copy import deepcopy
import logging

import numpy as np
from mindspore import context, set_seed, Tensor
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.train import DynamicLossScaleManager
from mindspore.nn import SGD

from src.utils import ValueInfo, update_config, init_log
from src.config import FasterRcnnConfig
from src.dataset import create_semisup_dataset
from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50 as Faster_Rcnn_Resnet
from src.network_define import StudentNetBurnInWithLoss, StudentNetBurnInDynamicTrainOneStep
from src.network_define import StudentNetBurnUpWithLoss, StudentNetBurnUpDynamicTrainOneStep
from src.lr_schedule import warmup_multi_step_lr, dynamic_lr


def train_combine():
    show_cfg()

    cfg = FasterRcnnConfig()

    set_seed(cfg.global_seed)
    device_id, rank_id, rank_size = set_device(cfg)
    logging.info("current device_id: %d, rank_id: %d, rank_size: %d", device_id, rank_id, rank_size)

    ####################################################################################
    # semisup dataset
    semisup_loader = create_semisup_dataset(cfg, is_training=True)

    ####################################################################################
    # net teacher
    net_teacher = Faster_Rcnn_Resnet(config=cfg)
    net_teacher = net_teacher.set_train(mode=False)

    if cfg.resume:
        cfg.filter_prefix = None
        if cfg.start_iter > cfg.burn_up_iter and cfg.pre_trained_teacher:
            param_dict = load_checkpoint(ckpt_file_name=cfg.pre_trained_teacher)
            param_not_load, _ = load_param_into_net(net_teacher, param_dict, strict_load=True)
            logging.info("load %s to net_teacher, params not load: %s", cfg.pre_trained_teacher, param_not_load)
        elif cfg.start_iter > cfg.burn_up_iter and not cfg.pre_trained_teacher:
            raise Exception("pre_trained_teacher ckpt is not set.")

    logging.info("net_teacher create ok.")

    ####################################################################################
    # net student
    net_student = Faster_Rcnn_Resnet(config=cfg)
    net_student = net_student.set_train(mode=True)

    if cfg.pre_trained:
        param_dict = load_checkpoint(ckpt_file_name=cfg.pre_trained,
                                     choice_func=lambda x: not x.startswith(tuple(cfg.filter_prefix)))
        param_not_load, _ = load_param_into_net(net_student, param_dict, strict_load=True)
        logging.info("load %s to net_student, params not load: %s", cfg.pre_trained, param_not_load)

    if cfg.device_target == "Ascend":
        net_student.to_float(mstype.float16)
        net_teacher.to_float(mstype.float16)

    lr_dict = {"consine": dynamic_lr, "step": warmup_multi_step_lr}
    lr_schedule = lr_dict.get(cfg.lr_schedule, warmup_multi_step_lr)
    lr = lr_schedule(cfg)
    logging.info("use %s lr", cfg.lr_schedule)
    opt = SGD(params=net_student.trainable_params(), learning_rate=Tensor(lr, mstype.float32), momentum=cfg.momentum,
              weight_decay=cfg.weight_decay)

    ####################################################################################

    net_student_burn_in_train, net_student_burn_up_train = create_student_train_model(net_student, cfg, opt)
    logging.info("net_student create ok.")

    logging.info("========================================")
    steps_per_epoch = semisup_loader.get_dataset_size()
    logging.info("total_steps: %d, steps_per_epoch: %d", cfg.max_iter, steps_per_epoch)
    logging.info("Processing, please wait a moment.")

    step_iter = cfg.start_iter
    start_time = time.perf_counter()
    keep_rate = 0.0
    for data in semisup_loader.create_dict_iterator(num_epochs=-1):

        label_data_q, label_data_k, label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums = \
            data["label_img_strong"], data["label_img_weak"], data["label_img_metas"], \
            data["label_gt_bboxes"], data["label_gt_labels"], data["label_gt_nums"]
        unlabel_data_q, unlabel_data_k, unlabel_img_metas, unlabel_gt_bboxes, unlabel_gt_labels, unlabel_gt_nums = \
            data["unlabel_img_strong"], data["unlabel_img_weak"], data["unlabel_img_metas"], \
            data["unlabel_gt_bboxes"], data["unlabel_gt_labels"], data["unlabel_gt_nums"]
        data_time = time.perf_counter() - start_time

        if step_iter < cfg.burn_up_iter:
            # burn-in stage (supervised training with labeled data)
            # all_label_data is: label_data_q + label_data_k
            loss = net_student_burn_in_train(label_data_q, label_data_k, label_img_metas,
                                             label_gt_bboxes, label_gt_labels, label_gt_nums)
        else:
            # step_iter = cfg.burn_up_iter: copy the the whole student model weight to teacher model
            # step_iter > cfg.burn_up_iter: use ema algorithm with student model to update teacher model
            update_teacher_net(net_teacher, net_student, keep_rate=keep_rate)
            keep_rate = cfg.ema_keep_rate

            # generate the pseudo-label using teacher model
            unlabel_img_metas_tmp = process_img_metas_for_inference(unlabel_img_metas)
            inference_output = net_teacher(unlabel_data_k, unlabel_img_metas_tmp,
                                           unlabel_gt_bboxes, unlabel_gt_labels, unlabel_gt_nums)
            pseudo_bboxes, pseudo_labels, pseudo_nums = pseudo_post_process(cfg, inference_output)

            # all_label_data is: label_data_q + label_data_k, all_unlabel_data is: unlabel_data_q
            loss = net_student_burn_up_train(label_data_q, label_data_k, label_img_metas,
                                             label_gt_bboxes, label_gt_labels, label_gt_nums,
                                             unlabel_data_q, unlabel_img_metas,
                                             pseudo_bboxes, pseudo_labels, pseudo_nums)

        iter_time = time.perf_counter() - start_time

        # update training infos
        if step_iter in (cfg.start_iter, cfg.burn_up_iter):
            time_infos = [ValueInfo("iter_time"), ValueInfo("data_time")]
            loss_infos = [ValueInfo("loss") for _ in range(len(loss))]

        for i in range(len(loss)):
            loss_infos[i].update(loss[i].asnumpy())
        time_infos[0].update(iter_time)
        time_infos[1].update(data_time)

        if step_iter % cfg.print_interval_iter == 0:
            if step_iter < cfg.burn_up_iter:
                logging.info("rank_id:%d, step: %d, avg loss: %.5f, rpn_cls_loss: %.5f, rpn_reg_loss: %.5f, "
                             "rcnn_cls_loss: %.5f, rcnn_reg_loss: %.5f, iter_time: %.3f, data_time: %.3f, "
                             "overflow: %f, scaling_sens: %f, lr: %.6f",
                             rank_id, step_iter, loss_infos[0].avg(), loss_infos[1].avg(), loss_infos[2].avg(),
                             loss_infos[3].avg(), loss_infos[4].avg(), time_infos[0].avg(), time_infos[1].avg(),
                             loss_infos[5].avg(), loss_infos[6].avg(), lr[step_iter])
            else:
                logging.info("rank_id:%d, step: %d, avg loss: %.5f, rpn_cls_loss: %.5f, "
                             "rpn_reg_loss: %.5f, rcnn_cls_loss: %.5f, rcnn_reg_loss: %.5f, "
                             "rpn_cls_loss_pseudo: %.5f, rpn_reg_loss_pseudo: %.5f, rcnn_cls_loss_pseudo: %.5f, "
                             "rcnn_reg_loss_pseudo: %.5f, iter_time: %.3f, data_time: %.5f, "
                             "overflow: %f, scaling_sens: %f, lr: %.6f",
                             rank_id, step_iter, loss_infos[0].avg(), loss_infos[1].avg(), loss_infos[2].avg(),
                             loss_infos[3].avg(), loss_infos[4].avg(), loss_infos[5].avg(), loss_infos[6].avg(),
                             loss_infos[7].avg(), loss_infos[8].avg(), time_infos[0].avg(), time_infos[1].avg(),
                             loss_infos[9].avg(), loss_infos[10].avg(), lr[step_iter])

        save_model(cfg, rank_id, step_iter, net_student, net_teacher)

        if step_iter == cfg.max_iter - 1:
            break

        step_iter += 1
        start_time = time.perf_counter()

    logging.info("end.")


def create_student_train_model(net_student, cfg, opt):
    # burn-in
    net_student_burn_in_with_loss = StudentNetBurnInWithLoss(net_student)
    loss_scale_manager_burn_in = DynamicLossScaleManager(scale_factor=2, scale_window=500)
    net_student_burn_in_train = StudentNetBurnInDynamicTrainOneStep(net_student_burn_in_with_loss, opt,
                                                                    loss_scale_manager_burn_in.get_update_cell())
    net_student_burn_in_train = net_student_burn_in_train.set_train(mode=True)

    # burn-up
    net_student_burn_up_with_loss = StudentNetBurnUpWithLoss(net_student, cfg.unsup_loss_weight)
    loss_scale_manager_burn_up = DynamicLossScaleManager(scale_factor=2, scale_window=500)
    net_student_burn_up_train = StudentNetBurnUpDynamicTrainOneStep(net_student_burn_up_with_loss, opt,
                                                                    loss_scale_manager_burn_up.get_update_cell())
    net_student_burn_up_train = net_student_burn_up_train.set_train(mode=True)
    return net_student_burn_in_train, net_student_burn_up_train


def save_model(cfg, rank_id, step_iter, net_student, net_teacher):
    if rank_id == 0 and cfg.save_checkpoint and step_iter >= cfg.burn_up_iter and \
            ((step_iter - cfg.start_iter) % cfg.save_checkpoint_interval == 0 or
             step_iter == cfg.burn_up_iter or step_iter == cfg.max_iter - 1):

        # save step model
        step_ckpt_name = "model_step_{}.ckpt"
        step_ckpt_file = os.path.join(cfg.save_checkpoint_path, step_ckpt_name.format(step_iter))
        save_checkpoint(net_student, step_ckpt_file)
        logging.info("save model to %s, at step=%s", step_ckpt_file, step_iter)

        # when in burn-up stage, need to save teacher model ckpt for mAP
        if step_iter >= cfg.burn_up_iter:
            teacher_step_ckpt_name = "teacher_{}".format(step_ckpt_name)
            teacher_step_ckpt_file = os.path.join(cfg.save_checkpoint_path,
                                                  teacher_step_ckpt_name.format(step_iter))
            save_checkpoint(net_teacher, teacher_step_ckpt_file)
            logging.info("save model teacher to %s, at step=%s", teacher_step_ckpt_file, step_iter)
            teacher_latest_ckpt_name = "model_0.ckpt"
            teacher_latest_ckpt_file = os.path.join(cfg.save_checkpoint_path, teacher_latest_ckpt_name)
            save_checkpoint(net_teacher, teacher_latest_ckpt_file)
            logging.info("save model teacher to %s, at step=%s", teacher_latest_ckpt_file, step_iter)


def show_cfg():
    logging.info("========================================")
    logging.info("Config contents:")
    for cfg_k, cfg_v in FasterRcnnConfig.__dict__.items():
        if not cfg_k.startswith("_"):
            logging.info("%s: %s", cfg_k, cfg_v)
    logging.info("========================================")


def set_device(cfg):
    device_id = int(os.getenv('DEVICE_ID', cfg.device_id))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=device_id)
    context.set_context(max_call_depth=6000)
    rank_id = 0
    rank_size = 1

    if cfg.run_distribute:
        if cfg.device_target == "Ascend":
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
            context.set_auto_parallel_context(device_num=rank_size, global_rank=rank_id, gradients_mean=True,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, parameter_broadcast=True)
        else:
            init("nccl")
            context.reset_auto_parallel_context()
            rank_id = get_rank()
            rank_size = get_group_size()
            context.set_auto_parallel_context(device_num=rank_size, global_rank=rank_id, gradients_mean=True,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, parameter_broadcast=True)
    return device_id, rank_id, rank_size


def update_teacher_net(teacher, student, keep_rate=0.996):
    """Update the EMA parameters"""
    params_teacher = teacher.parameters_dict()
    params_student = student.parameters_dict()

    for key, param in params_teacher.items():
        if key in params_student.keys():
            new_value = (1.0 - keep_rate) * params_student[key].data.asnumpy().copy() + \
                        keep_rate * param.data.asnumpy().copy()
            param.set_data(Tensor(deepcopy(new_value)))
        else:
            raise Exception("teacher model key [{}] is not found in student model.".format(key))


def process_img_metas_for_inference(img_metas):
    if img_metas.shape[1] != 3:
        raise Exception("img_metas shape must be ({}, 3), but get ({}, {}) instead."
                        .format(img_metas.shape[0], img_metas.shape[0], img_metas.shape[1]))
    img_metas_tmp = img_metas.asnumpy()
    img_metas_tmp = np.column_stack((img_metas_tmp, img_metas_tmp[:, 2:3]))
    img_metas_tmp = np.asarray(img_metas_tmp, dtype=np.float32)
    img_metas_tmp = Tensor.from_numpy(img_metas_tmp)
    return img_metas_tmp


def pseudo_post_process(cfg, inference_output):
    outputs = []
    max_num = 128

    all_bbox = inference_output[0]
    all_label = inference_output[1]
    all_mask = inference_output[2]

    for j in range(cfg.test_batch_size):
        all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
        all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
        all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]

        if all_bboxes_tmp_mask.shape[0] > max_num:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
            inds = inds[:max_num]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]

        outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, cfg.num_classes)
        outputs.append(outputs_tmp)

    pseudo_bboxes, pseudo_labels, pseudo_nums = process_pseudo_label(cfg, outputs)
    pseudo_bboxes = Tensor.from_numpy(pseudo_bboxes)
    pseudo_labels = Tensor.from_numpy(pseudo_labels)
    pseudo_nums = Tensor.from_numpy(pseudo_nums)

    return pseudo_bboxes, pseudo_labels, pseudo_nums


def process_pseudo_label(cfg, proposals):
    pseudo_bboxes = []
    pseudo_labels = []
    pseudo_isvalid = []
    pad_max_number = 128
    for i in range(cfg.test_batch_size):
        proposals_per_batch = proposals[i]
        pseudo_bboxes_per_batch = np.array(np.zeros((0, 4), dtype=np.float32))
        pseudo_labels_per_batch = np.array(np.zeros(0, dtype=np.float32))
        for cls, proposals_per_cls in enumerate(proposals_per_batch):
            if proposals_per_cls.size:
                for proposal in proposals_per_cls:
                    if proposal[4] > cfg.bbox_threshold:
                        pseudo_bboxes_per_batch = np.row_stack((pseudo_bboxes_per_batch, proposal[:4]))
                        pseudo_labels_per_batch = np.append(pseudo_labels_per_batch, cls + 1)

        pseudo_bboxes_per_batch_new = np.pad(pseudo_bboxes_per_batch,
                                             ((0, pad_max_number - pseudo_bboxes_per_batch.shape[0]), (0, 0)),
                                             mode="constant",
                                             constant_values=0)
        pseudo_labels_per_batch_new = np.pad(pseudo_labels_per_batch,
                                             ((0, pad_max_number - pseudo_labels_per_batch.shape[0])),
                                             mode="constant",
                                             constant_values=-1)
        pseudo_isvalid_per_batch = np.ones(pseudo_labels_per_batch.shape[0], dtype=np.int32)
        pseudo_isvalid_per_batch_new = np.pad(pseudo_isvalid_per_batch,
                                              ((0, pad_max_number - pseudo_isvalid_per_batch.shape[0])),
                                              mode="constant",
                                              constant_values=0)

        pseudo_bboxes.append(pseudo_bboxes_per_batch_new)
        pseudo_labels.append(pseudo_labels_per_batch_new)
        pseudo_isvalid.append(pseudo_isvalid_per_batch_new)

    pseudo_bboxes = np.array(pseudo_bboxes).astype(np.float32)
    pseudo_labels = np.array(pseudo_labels).astype(np.int32)
    pseudo_isvalid = np.array(pseudo_isvalid).astype(np.bool)

    return pseudo_bboxes, pseudo_labels, pseudo_isvalid


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5), 5 column for (x1, y1, x2, y2, score)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
    return result


if __name__ == '__main__':
    init_log()
    update_config()
    train_combine()
