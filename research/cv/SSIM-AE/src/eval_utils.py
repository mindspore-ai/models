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

import os
from decimal import Decimal
import numpy as np
import cv2
from mindspore import nn
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from src.utils import read_img, get_img_names


def calculate_TP_TN_FP_FN_pixel(ground_truth, predicted_mask):
    """Calculate pixel level confusion matrix, return TP, TN, FP, FN."""
    TP = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 1))
    TN = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 0))
    FP = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 1))
    FN = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 0))
    return TP, TN, FP, FN


def calculate_TP_TN_FP_FN_image(ground_truth, predicted_mask):
    """Calculate image level confusion matrix, return TP, TN, FP, FN."""
    TP, TN, FP, FN = 0, 0, 0, 0
    # ground_truth is good
    if np.all(~ground_truth.astype(np.bool)):
        # if predicted greater than 1, it will predict to defect.
        if np.any(predicted_mask.astype(np.bool)):
            FN = 1
        else:
            TP = 1
    # the ground is defect image
    else:
        if np.any(predicted_mask.astype(np.bool)):
            TN = 1
        else:
            FP = 1
    return TP, TN, FP, FN


def round_acc(acc):
    acc_around = Decimal(acc).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
    return acc_around


def cal_metric(pred_list, gt_list, image_level=True):
    """Calculate confusion matrix, return TP, TN, FP, FN."""
    OK_1, OK_2 = 0, 0
    NOK_1, NOK_2 = 0, 0
    ACC_1, ACC_2 = 0, 0
    preds, gts = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        predict = (read_img(pred_path, True) > 0).astype(int)
        preds.append(predict)
        # If it's a positive sample directory and there is no GT label, create one.
        if gt_path == "good":
            gt = np.zeros(predict.shape, predict.dtype)
        else:
            gt = read_img(gt_path, True)
            gt = (cv2.resize(gt, predict.shape) > 0).astype(int)
        gts.append(gt)
        if image_level:
            TP, TN, FP, FN = calculate_TP_TN_FP_FN_image(ground_truth=gt, predicted_mask=predict)
        else:
            TP, TN, FP, FN = calculate_TP_TN_FP_FN_pixel(ground_truth=gt, predicted_mask=predict)
        OK_1 += TP
        OK_2 += TP + FN
        NOK_1 += TN
        NOK_2 += FP + TN
        ACC_1 += TP + TN
        ACC_2 += TP + TN + FP + FN
    if image_level:
        print(
            "ok: {}, nok: {}, avg: {}".format(
                round_acc(OK_1 / OK_2), round_acc(NOK_1 / NOK_2), round_acc(ACC_1 / ACC_2)
            )
        )
        return ACC_1 / ACC_2
    y_pred = np.array(preds).flatten()
    y = np.array(gts).flatten()
    metric = nn.ROC(pos_label=1)
    metric.clear()
    metric.update(y_pred, y)
    fpr, tpr, _ = metric.eval()
    auc = nn.auc(fpr, tpr)
    print(
        "AUC: {}, ok: {}, nok: {}, avg: {}".format(
            round_acc(auc), round_acc(OK_1 / OK_2), round_acc(NOK_1 / NOK_2), round_acc(ACC_1 / ACC_2)
        )
    )
    return auc


def apply_eval(cfg):
    file_names = get_img_names(cfg.save_dir, "_residual.png")
    pred_list, gt_list = [], []
    for img_name in file_names:
        pred_path = os.path.join(cfg.save_dir, img_name + "_residual.png")
        # t = img_name.split("_")
        # img_name = os.path.join(t[0], t[1])
        gt_path = os.path.join(cfg.gt_dir, img_name + cfg.mask_suffix)
        if not os.path.exists(gt_path):
            if "good" in gt_path:
                gt_path = "good"
            else:
                print(pred_path, "not found gt!")
                continue
        pred_list.append(pred_path)
        gt_list.append(gt_path)
    auc = cal_metric(pred_list, gt_list, cfg.image_level)
    return auc


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        config (config object): evaluation parameters' configure object
        net (Cell): evaluation network.
        get_results (function): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        best_ckpt_name (str): best checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(config, net, get_results, apply_eval)
    """

    def __init__(self, config, net, get_results, ssim_ae_eval):
        super(EvalCallBack, self).__init__()
        self.get_results = get_results
        self.ssim_ae_eval = ssim_ae_eval
        self.config = config
        self.net = net
        self.best_epoch = 0
        self.best_res = 0
        self.best_ckpt_path = "./checkpoint"

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        current_epoch = cb_params.cur_epoch_num
        res = 0
        if current_epoch >= self.config.start_epoch and (
            current_epoch % self.config.interval == 0 or current_epoch in self.config.eval_epochs
        ):
            ori_save_dir = self.config.save_dir
            self.config.save_dir = os.path.join(self.config.save_dir, str(current_epoch))
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            self.net.set_train(False)
            self.get_results(self.config, self.net)
            self.net.set_train(True)
            print("Generate results at", self.config.save_dir)
            res = self.ssim_ae_eval(self.config)
            self.config.save_dir = ori_save_dir
            self.config.ssim_threshold = -1
            self.config.l1_threshold = -1

        if res > self.best_res:
            self.best_epoch = current_epoch
            self.best_res = res

            if os.path.exists(os.path.join(self.best_ckpt_path, "best.ckpt")):
                os.remove(os.path.join(self.best_ckpt_path, "best.ckpt"))

            os.makedirs(self.best_ckpt_path, exist_ok=True)
            save_checkpoint(cb_params.train_network, os.path.join(self.best_ckpt_path, "best.ckpt"))

            print("update best result: {} in the {} th epoch".format(self.best_res, self.best_epoch), flush=True)

    def end(self, run_context):
        print("End training the best {0} epoch is {1}".format(self.best_res, self.best_epoch), flush=True)
