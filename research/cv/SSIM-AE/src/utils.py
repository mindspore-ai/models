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
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import morphology
import mindspore as ms
from mindspore import Tensor


def read_img(img_path, grayscale):
    """If grayscale is True, return grayscale image, else BGR image"""
    if grayscale:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    return im


def get_file_list(path, suffix=".png"):
    """Get the list of all file names with `suffix` suffix in this `path`."""
    file_list = []
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isdir(file_path):
            file_list.extend(get_file_list(file_path, suffix))
        elif file_path.endswith(suffix):
            file_list.append(file_path)
    return file_list


def get_img_names(dir_path, suffix):
    """Get the list of all file names with `suffix` suffix but delete the `suffix` in this `path`."""
    file_list = get_file_list(dir_path, suffix)
    img_names = []
    dir_path_len = len(dir_path) if dir_path[-1] == "/" else len(dir_path) + 1
    suffix_len = len(suffix)
    for file in file_list:
        img_names.append(file[dir_path_len:-suffix_len])
    return img_names


def get_patch(image, new_size, stride):
    """Get encode array when img size is different from mask size."""
    height, weighe = image.shape[:2]
    i, j = new_size, new_size
    patch = []
    while i <= height:
        while j <= weighe:
            patch.append(image[i - new_size : i, j - new_size : j])
            j += stride
        j = new_size
        i += stride
    return np.array(patch)


def patch2img(patches, im_size, patch_size, stride):
    """Get decode array when img size is different from mask size."""
    img = np.zeros((im_size, im_size, patches.shape[3] + 1))
    i, j = patch_size, patch_size
    k = 0
    while i <= im_size:
        while j <= im_size:
            img[i - patch_size : i, j - patch_size : j, :-1] += patches[k]
            img[i - patch_size : i, j - patch_size : j, -1] += np.ones((patch_size, patch_size))
            k += 1
            j += stride
        j = patch_size
        i += stride
    mask = np.repeat(img[:, :, -1][..., np.newaxis], patches.shape[3], 2)
    img = img[:, :, :-1] / mask
    return img


def get_residual_map(img_path, cfg, auto_encoder):
    """Get reconstructed image, ssim residual image and l1 residual image."""
    test_img = read_img(img_path, cfg.grayscale)

    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size) // 2
        test_img = test_img[tmp : tmp + cfg.mask_size, tmp : tmp + cfg.mask_size]
    test_img_ = test_img / 255.0
    if test_img.shape[:2] == (cfg.crop_size, cfg.crop_size):
        if cfg.grayscale:
            patches = np.expand_dims(test_img_, axis=(0, 1))
        else:
            patches = np.expand_dims(test_img_, axis=0)
            patches = np.transpose(patches, (0, 3, 1, 2))
        decoded_img = auto_encoder(Tensor(patches, ms.float32)).asnumpy()
        decoded_img = np.transpose(decoded_img, (0, 2, 3, 1))
    else:
        patches = get_patch(test_img_, cfg.crop_size, cfg.stride)
        if cfg.grayscale:
            patches = np.expand_dims(patches, 1)
        else:
            patches = np.transpose(patches, (0, 3, 1, 2))
        patches = auto_encoder(Tensor(patches, ms.float32)).asnumpy()
        patches = np.transpose(patches, (0, 2, 3, 1))
        decoded_img = patch2img(patches, cfg.im_resize, cfg.crop_size, cfg.stride)

    rec_img = np.reshape((decoded_img * 255.0).astype(np.uint8), test_img.shape)
    if cfg.image_level:
        if cfg.grayscale:
            ssim_residual_map = 1 - ssim(test_img, rec_img, win_size=11, full=True)[1]
            l1_residual_map = np.abs(test_img / 255.0 - rec_img / 255.0)
        else:
            ssim_residual_map = ssim(test_img, rec_img, win_size=11, full=True, multichannel=True)[1]
            ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
            l1_residual_map = np.mean(np.abs(test_img / 255.0 - rec_img / 255.0), axis=2)
    else:
        if cfg.grayscale:
            ssim_residual_map = (
                1
                - ssim(
                    test_img,
                    rec_img,
                    win_size=11,
                    data_range=1,
                    gradient=False,
                    full=True,
                    gaussian_weights=True,
                    sigma=7,
                )[1]
            )
            l1_residual_map = np.abs(test_img / 255.0 - rec_img / 255.0)
        else:
            ssim_residual_map = ssim(
                test_img,
                rec_img,
                win_size=11,
                data_range=1,
                gradient=False,
                full=True,
                channel_axis=2,
                gaussian_weights=True,
                sigma=7,
            )[1]
            ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
            l1_residual_map = np.mean(np.abs(test_img / 255.0 - rec_img / 255.0), axis=2)

    return test_img, rec_img, ssim_residual_map, l1_residual_map


def get_threshold(cfg, auto_encoder):
    """Statistics of SSIM threshold using training set."""
    print("estimating threshold...")
    valid_good_list = sorted(get_file_list(cfg.train_data_dir, cfg.img_suffix))
    num_valid_data = int(np.ceil(len(valid_good_list) * 0.2))
    total_rec_ssim, total_rec_l1 = [], []
    for img_path in valid_good_list[-num_valid_data:]:
        _, _, ssim_residual_map, l1_residual_map = get_residual_map(img_path, cfg, auto_encoder)
        total_rec_ssim.append(ssim_residual_map)
        total_rec_l1.append(l1_residual_map)
    total_rec_ssim = np.array(total_rec_ssim)
    total_rec_l1 = np.array(total_rec_l1)
    ssim_threshold = float(np.percentile(total_rec_ssim, cfg.percent))
    l1_threshold = float(np.percentile(total_rec_l1, cfg.percent))
    if cfg.ssim_threshold < 0:
        cfg.ssim_threshold = ssim_threshold
    if cfg.l1_threshold < 0:
        cfg.l1_threshold = l1_threshold
    print("ssim_threshold: {}, l1_threshold: {}".format(cfg.ssim_threshold, cfg.l1_threshold))


def bg_mask(img, value, mode, grayscale):
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, value, 255, mode)

    def FillHole(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out

    thresh = FillHole(thresh)
    if isinstance(thresh, int):
        return np.ones(img.shape)
    mask_ = np.ones(thresh.shape)
    mask_[np.where(thresh <= 127)] = 0
    return mask_


def get_depressing_mask(cfg):
    depr_mask = np.ones((cfg.mask_size, cfg.mask_size)) * 0.2
    depr_mask[5 : cfg.mask_size - 5, 5 : cfg.mask_size - 5] = 1
    cfg.depr_mask = depr_mask


def set_img_color(img, predict_mask, weight_foreground, grayscale):
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    origin = img
    img[np.where(predict_mask == 255)] = (0, 0, 255)
    cv2.addWeighted(img, weight_foreground, origin, (1 - weight_foreground), 0, img)
    return img


def get_results(cfg, auto_encoder, test_list=None):
    get_depressing_mask(cfg)
    if cfg.ssim_threshold < 0 or cfg.l1_threshold < 0:
        get_threshold(cfg, auto_encoder)
    if test_list is None:
        test_list = get_file_list(cfg.test_dir, cfg.img_suffix)
    test_path_s = len(cfg.test_dir)
    if cfg.test_dir[-1] != "/":
        test_path_s += 1
    for img_path in test_list:
        test_img, rec_img, ssim_residual_map, l1_residual_map = get_residual_map(img_path, cfg, auto_encoder)
        ssim_residual_map *= cfg.depr_mask
        mask = np.zeros((cfg.mask_size, cfg.mask_size))
        mask[ssim_residual_map > cfg.ssim_threshold] = 1
        mask[l1_residual_map > cfg.l1_threshold] = 1
        if cfg.bg_mask == "B":
            bg_m = bg_mask(test_img.copy(), 50, cv2.THRESH_BINARY, cfg.grayscale)
            mask *= bg_m
        elif cfg.bg_mask == "W":
            bg_m = bg_mask(test_img.copy(), 200, cv2.THRESH_BINARY_INV, cfg.grayscale)
            mask *= bg_m
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask = mask * 255
        file_name, _ = os.path.splitext(img_path)
        img_name = file_name[test_path_s:]
        vis = set_img_color(test_img.copy(), mask, weight_foreground=0.3, grayscale=cfg.grayscale)
        base_dir = os.path.dirname(os.path.join(cfg.save_dir, img_name))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_residual.png"), mask)
        cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_origin.png"), test_img)
        cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_rec.png"), rec_img)
        cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_visual.png"), vis)
