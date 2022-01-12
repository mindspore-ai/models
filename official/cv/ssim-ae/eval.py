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
import os
from glob import glob
import numpy as np
import cv2
from skimage import morphology
from skimage.metrics import structural_similarity as ssim

import mindspore as ms
from mindspore import context, Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net

from src.dataset import get_patch, patch2img
from src.network import AutoEncoder
from model_utils.device_adapter import get_device_id
from model_utils.options import Options

cfg = Options().parse()

context.set_context(mode=context.GRAPH_MODE
                    , device_target=cfg["device_target"], device_id=get_device_id())

autoencoder = AutoEncoder(cfg)


def read_img(img_path, grayscale):
    if grayscale:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    return im


if cfg["model_arts"]:
    import moxing as mox

    mox.file.copy_parallel(src_url=cfg["checkpoint_url"], dst_url=cfg["ckpt_file"])
    ckpt_path = cfg["ckpt_file"]
else:
    ckpt_path = cfg["checkpoint_path"]

param_dict = load_checkpoint(ckpt_path
                             , net=autoencoder)
load_param_into_net(autoencoder, param_dict)
autoencoder.set_train(False)

im_resize = cfg["data_augment"]["im_resize"]
crop_size = cfg["data_augment"]["crop_size"]


def get_residual_map(img_path):
    test_img = read_img(img_path, cfg["grayscale"])
    if test_img.shape[:2] != (im_resize, im_resize):
        test_img = cv2.resize(test_img, (im_resize, im_resize)) / 255.0

    if im_resize != cfg["mask_size"]:
        tmp = (im_resize - cfg["mask_size"]) // 2
        test_img_ = test_img[tmp:tmp + cfg["mask_size"], tmp:tmp + cfg["mask_size"]]
    else:
        test_img_ = test_img

    if test_img_.shape[:2] == (crop_size, crop_size):
        if cfg["grayscale"]:
            patches = np.expand_dims(test_img_, axis=(0, 1))
        else:
            patches = np.expand_dims(test_img_, axis=0)
            patches = np.transpose(patches, (0, 3, 1, 2))
        decoded_img = autoencoder(Tensor(patches, ms.float32))
        squeeze = ops.Squeeze()
        rec_img = squeeze(decoded_img)
        rec_img = rec_img.asnumpy()
        if not cfg["grayscale"]:
            rec_img = np.transpose(rec_img, (1, 2, 0))
        rec_img = (rec_img * 255).astype('uint8')
    else:
        patches = get_patch(test_img_, crop_size, cfg["stride"])
        if cfg["grayscale"]:
            patches = np.expand_dims(patches, 1)
        else:
            patches = np.transpose(patches, (0, 3, 1, 2))
        patches = autoencoder(Tensor(patches, ms.float32))
        patches = patches.asnumpy()
        patches = np.transpose(patches, (0, 2, 3, 1))
        rec_img = patch2img(patches, im_resize, crop_size, cfg["stride"])
        rec_img = np.reshape((rec_img * 255).astype('uint8'), test_img_.shape)

    img1 = test_img_.astype("float32")
    img2 = rec_img.astype("float32") / 255.0

    if cfg["grayscale"]:
        _, diff = ssim(img1, img2, win_size=11
                       , data_range=1, gradient=False, full=True, gaussian_weights=True, sigma=10)
        loss = 1 - diff
    else:
        rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)
        _, diff = ssim(img1, img2, win_size=11, data_range=1, gradient=False, full=True, channel_axis=2
                       , gaussian_weights=True, sigma=10)
        loss = 1 - np.mean(diff, axis=2)

    loss *= 1 / 2
    test_img_ = (test_img_ * 255).astype('uint8')
    return test_img_, rec_img, loss


def get_threshold(d_path):
    print('estimating threshold...')
    train_path = d_path + "/train"
    sub = os.listdir(train_path)
    if os.path.isdir(train_path + "/" + sub[0]):
        valid_good_list = sorted(glob(train_path + "/" + sub[0] + '/*png'))
    else:
        valid_good_list = sorted(glob(train_path + '/*png'))

    num_valid_data = int(np.ceil(len(valid_good_list) * 0.2))
    total_rec_ssim = []
    for img_path in valid_good_list[-num_valid_data:]:
        _, _, ssim_residual_map = get_residual_map(img_path)
        total_rec_ssim.append(ssim_residual_map)
    total_rec_ssim = np.array(total_rec_ssim)
    ssim_threshold = float(np.percentile(total_rec_ssim, cfg["percent"]))
    print('ssim_threshold: %f' % ssim_threshold)
    if not cfg["ssim_threshold"]:
        cfg["ssim_threshold"] = ssim_threshold


def calculate_TP_TN_FP_FN(ground_truth, predicted_mask):
    TP = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 1))
    TN = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 0))
    FP = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 1))
    FN = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 0))
    return TP, TN, FP, FN


def get_depressing_mask():
    depr_mask = np.ones((cfg["mask_size"], cfg["mask_size"])) * 0.2
    depr_mask[5:cfg["mask_size"] - 5, 5:cfg["mask_size"] - 5] = 1
    cfg["depr_mask"] = depr_mask


def get_results(file_list):
    pred_list, gt_list = [], []
    ok_1, ok_2 = 0, 0
    nok_1, nok_2 = 0, 0
    Acc_1, Acc_2 = 0, 0
    for img_path in file_list:
        test_img, _, loss = get_residual_map(img_path)
        pred_list.append(loss)

        img_name = img_path.split('/')[-1][:-4]
        gt_path = img_path.replace("test", "ground_truth")
        gt_path = gt_path[:-7]
        sub = os.listdir(gt_path)

        if os.path.isdir(gt_path + sub[0]):
            gt_path = gt_path + sub[0] + "/"

        gt_path = glob(gt_path + "*png")
        for item in gt_path:
            gt_name = item.split('/')[-1][0:3]
            if img_name == gt_name:
                gt_path = item
                break
        gt_img = cv2.imread(gt_path, 0)
        gt_img = cv2.resize(gt_img, test_img.shape[:2])
        gt_img = (gt_img > 0).astype(int)
        gt_list.append(gt_img)

        loss *= cfg["depr_mask"]
        mask = np.zeros(gt_img.shape, dtype=float)
        mask[loss > cfg["ssim_threshold"]] = 1
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)

        TP, TN, FP, FN = calculate_TP_TN_FP_FN(ground_truth=gt_img, predicted_mask=mask)

        ok_1 += TP
        ok_2 += (TP + FN)
        nok_1 += TN
        nok_2 += (FP + TN)
        Acc_1 += (TP + TN)
        Acc_2 += (TP + TN + FP + FN)

    pred_list = np.array(pred_list)
    gt_list = np.array(gt_list)

    return pred_list, gt_list, ok_1, ok_2, nok_1, nok_2, Acc_1, Acc_2


if __name__ == '__main__':

    test_path = cfg["test_dir"]
    data_path = test_path[:-5]
    get_depressing_mask()
    sub_folder = os.listdir(test_path)
    if os.path.isdir(test_path + "/" + sub_folder[0]):
        OK_1, OK_2 = 0, 0
        NOK_1, NOK_2 = 0, 0
        ACC_1, ACC_2 = 0, 0
        if not cfg["ssim_threshold"]:
            get_threshold(data_path)
        for k in sub_folder:
            if k != "good":
                test_list = glob(data_path + '/test/' + k + '/*')
                _, _, a1, a2, b1, b2, c1, c2 = get_results(test_list)
                OK_1 += a1
                OK_2 += a2
                NOK_1 += b1
                NOK_2 += b2
                ACC_1 += c1
                ACC_2 += c2
        print("ok:", OK_1 / OK_2)
        print("nok:", NOK_1 / NOK_2)
        print("acc:", ACC_1 / ACC_2)
    else:
        test_list = glob(data_path + "/test" + '/*png')
        y_pred, y, _, _, _, _, _, _, = get_results(test_list)
        y_pred = y_pred.flatten()
        y = y.flatten()

        metric = nn.ROC(pos_label=1)
        metric.clear()
        metric.update(y_pred, y)
        fpr, tpr, thre = metric.eval()
        output = nn.auc(fpr, tpr)
        print("AUC:", output)
