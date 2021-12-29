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
import shutil
import os
import argparse
import numpy as np
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from src.dataset import patch2img
from model_utils.options import Options_310
import cv2
import mindspore.nn as nn


def read_img(img_path, grayscale):
    if grayscale:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    return im


def get_depressing_mask():
    depr_mask = np.ones((cfg["mask_size"], cfg["mask_size"])) * 0.2
    depr_mask[5:cfg["mask_size"] - 5, 5:cfg["mask_size"] - 5] = 1
    cfg["depr_mask"] = depr_mask


def get_list(path):
    res_list = []
    file_list = []
    dirs = os.listdir(path)
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)
    for file_dir in file_list:
        gt_img = cv2.imread(path + file_dir, 0)
        if im_resize != cfg["mask_size"]:
            gt_img = cv2.resize(gt_img, (cfg["mask_size"], cfg["mask_size"]))
        else:

            gt_img = cv2.resize(gt_img, (im_resize, im_resize))
        gt_img = (gt_img > 0).astype(int)
        res_list.append(gt_img)
    return res_list


def calculate_TP_TN_FP_FN(ground_truth, predicted_mask):
    TP = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 1))
    TN = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 0))
    FP = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 1))
    FN = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 0))
    return TP, TN, FP, FN


def make_clean(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def copy_to(path):
    sub_fold = os.listdir(path)
    target_path = path + "/all_test/"

    if not os.path.exists(target_path):
        os.mkdir(target_path)
    else:
        shutil.rmtree(target_path)
        os.mkdir(target_path)
    sub_fold = sorted(sub_fold)
    for i, sub_folder_name in enumerate(sub_fold):
        if sub_folder_name not in ('all_test', 'good'):
            imgName_list = os.listdir(path + sub_folder_name)
            imgName_list = sorted(imgName_list)
            for x in imgName_list:
                image_path = path + sub_folder_name + "/" + x

                shutil.copy(image_path, target_path + str(i) + "_" + x)


def postprocess(sub=False):
    ok_1 = 0
    ok_2 = 0
    nok_1 = 0
    nok_2 = 0
    Acc_1 = 0
    Acc_2 = 0
    rst_path = args.result_path
    gt_path = args.gt_path


    if sub:
        groundtruth_path = gt_path + "/ground_truth/all_test/"
        tst_path = gt_path + "/test/all_test/"
    else:
        groundtruth_path = gt_path + "/ground_truth/defective/"
        tst_path = gt_path + "/test/"

    gt_list = get_list(groundtruth_path)
    pred_list = []
    imagePath_list = []
    dirs = os.listdir(tst_path)
    for file in dirs:
        imagePath_list.append(file)
    imagePath_list = sorted(imagePath_list)

    patch_num = int(pow(((im_resize - crop_size) / cfg["stride"]) + 1, 2))
    for i in range(int(len(os.listdir(rst_path)) / patch_num)):
        img_shape = im_resize
        if im_resize != cfg["mask_size"]:
            img_shape = cfg["mask_size"]
        if crop_size == img_shape:
            file_name = os.path.join(rst_path, "AE_SSIM_" + str(i) + "_patch_0_0.bin")
            output = np.fromfile(file_name, np.float32).reshape(input_channel, crop_size, crop_size)
            output = output.transpose((1, 2, 0))
            if cfg["grayscale"]:
                output = np.reshape((output * 255).astype('uint8'), (im_resize, im_resize))
            else:
                output = (output * 255).astype('uint8')
        else:
            patches = []
            for j in range(0, patch_num):
                file_name = os.path.join(rst_path, "AE_SSIM_" + str(i) + "_patch_" + str(j) + "_0.bin")
                patch = np.fromfile(file_name, np.float32).reshape(input_channel, crop_size, crop_size)

                patches.append(patch.transpose((1, 2, 0)))
            patches = np.asarray(patches)
            output = patch2img(patches, im_resize, crop_size, cfg["stride"])

            if cfg["grayscale"]:
                output = np.reshape((output * 255).astype('uint8'), (img_shape, img_shape))
            else:
                output = np.reshape((output * 255).astype('uint8'), (img_shape, img_shape, 3))


        print("=======image", i + 1, "saved success=======", imagePath_list[i])

        # cal SSIM
        img1 = read_img(tst_path + imagePath_list[i], cfg["grayscale"])
        img1 = cv2.resize(img1, (im_resize, im_resize)) / 255.0
        if im_resize != cfg["mask_size"]:
            tmp = (im_resize - cfg["mask_size"]) // 2
            img1 = img1[tmp:tmp + cfg["mask_size"], tmp:tmp + cfg["mask_size"]]
        img1 = img1.astype("float32")
        img2 = output.astype("float32") / 255.0  # recon image

        if cfg["grayscale"]:
            _, diff = ssim(img1, img2, win_size=11
                           , data_range=1, gradient=False, full=True, gaussian_weights=True, sigma=10)
            loss = 1 - diff
        else:

            _, diff = ssim(img1, img2, win_size=11, data_range=1, gradient=False, full=True, channel_axis=2
                           , gaussian_weights=True, sigma=10, multichannel=True)
            loss = 1 - np.mean(diff, axis=2)

        loss *= 1 / 2
        gt_img = gt_list[i]

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

        pred_list.append(loss)

    ok = ok_1 / ok_2
    nok = nok_1 / nok_2
    Acc = Acc_1 / Acc_2
    print("ok:", ok)
    print("nok:", nok)
    print("Acc:", Acc)

    pred_list = np.array(pred_list)
    gt_list = np.array(gt_list)

    gt_list = gt_list.flatten()
    pred_list = pred_list.flatten()

    metric = nn.ROC(pos_label=1)
    metric.clear()
    metric.update(pred_list, gt_list)
    fpr, tpr, _ = metric.eval()
    output = nn.auc(fpr, tpr)
    print("AUC:", output)


parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--result_path', type=str, default='./data/result_Files/', help='eval data dir')
parser.add_argument('--gt_path', type=str, default='./datasets/metal_nut/', help='result path')
parser.add_argument('--save_path', type=str, default='./data/save_img/', help='result path')
args = parser.parse_args()
cfg = Options_310().opt
input_channel = 1 if cfg["grayscale"] else 3
im_resize = cfg["data_augment"]["im_resize"]
crop_size = cfg["data_augment"]["crop_size"]
if __name__ == '__main__':
    data_path = args.gt_path
    test_path = data_path + "/test/"
    sub_folder = os.listdir(test_path)
    get_depressing_mask()
    if os.path.isdir(test_path + "/" + sub_folder[0]):
        copy_to(data_path + "/test/")
        copy_to(data_path + "/ground_truth/")
        postprocess(True)
    else:
        postprocess()
