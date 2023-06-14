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

"""Evaluation for Semantic Human Matting"""
import os
import time

import cv2
import numpy as np

from mindspore import dtype as mstype
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

import src.model.network as network


def safe_makedirs(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def safe_modify_file_name(file_name):
    if not os.path.exists(file_name):
        if "jpg" in file_name:
            return file_name.replace("jpg", "png")
        return file_name.replace("png", "jpg")

    return file_name


def seg_process(cfg, image, image_gt, net):
    """Perform inference and calculate metric"""
    origin_h, origin_w, _ = image.shape

    # resize and normalize
    image_resize = cv2.resize(image, (cfg["size"], cfg["size"]), interpolation=cv2.INTER_CUBIC)
    image_resize = (
        image_resize
        - (
            104.0,
            112.0,
            121.0,
        )
    ) / 255.0

    # construct input tensor
    x = np.expand_dims(image_resize, axis=3)
    inputs = np.transpose(x, (3, 2, 0, 1))

    # inference
    trimap, alpha = net(Tensor(inputs, dtype=mstype.float32))

    # generate mask
    trimap_np = trimap[0, 0, :, :].asnumpy()
    trimap_np = cv2.resize(trimap_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    mask_result = np.multiply(trimap_np[..., np.newaxis], image)

    trimap_1 = mask_result.copy()
    mask_result[trimap_1 < 10] = 255
    mask_result[trimap_1 >= 10] = 0

    # generate foreground image
    alpha_np = alpha[0, 0, :, :].asnumpy()
    alpha_fg = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    fg = np.multiply(alpha_fg[..., np.newaxis], image)

    # generate metric Sad (original image size)
    image_gt = image_gt[:, :, 0]
    image_gt = image_gt.astype(np.float64) / 255
    sad = np.abs(alpha_fg - image_gt).sum() / 1000

    return mask_result, fg, sad


def camera_seg(cfg, net):
    """Perform inference, save result and calculate metric"""
    test_pic_path = cfg["test_pic_path"]
    output_path = cfg["output_path"]
    safe_makedirs(output_path)

    f_log = open(os.path.join(output_path, "log.txt"), "w")

    time_0 = time.time()
    file_test_list = os.path.join(test_pic_path, "test", "test.txt")
    list_sad = list()
    with open(file_test_list) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            img_clip = os.path.join(test_pic_path, "test", "clip_img", line.replace("matting", "clip"))
            img_alpha = os.path.join(test_pic_path, "test", "alpha", line)
            img_clip = safe_modify_file_name(img_clip)
            img_alpha = safe_modify_file_name(img_alpha)

            path_save = os.path.join(output_path, "clip_img_rst", os.path.split(line)[0].replace("matting", "clip"))
            safe_makedirs(path_save)

            img_src = cv2.imread(img_clip)
            img_gt = cv2.imread(img_alpha)
            mask_result, fg, sad = seg_process(cfg, img_src, img_gt, net)

            file_name = os.path.split(line)[1]
            cv2.imwrite(os.path.join(path_save, file_name), mask_result)
            cv2.imwrite(os.path.join(path_save, file_name.split(".")[0] + "_fg.jpg"), fg)

            log = "{} sad: {}".format(os.path.join(path_save, file_name), sad)
            print(log)
            f_log.write(log + "\n")
            list_sad.append(sad)
    log = "Total time: {}, ave_sad: {}".format(time.time() - time_0, np.mean(list_sad))
    print(log)
    f_log.write(log)


def run_test(cfg):
    device_id = int(os.getenv("DEVICE_ID", "0"))
    print("device_id: {}".format(device_id))
    context.set_context(
        mode=context.GRAPH_MODE,
        device_id=device_id,
        device_target=cfg["device_target"],
        reserve_class_name_in_scope=False,
    )

    net = network.net()
    print(cfg["model"])

    param_dict = load_checkpoint(cfg["model"])
    load_param_into_net(net, param_dict)

    net.set_train(False)
    camera_seg(cfg, net)


if __name__ == "__main__":
    from src.config import get_args, get_config_from_yaml

    run_test(get_config_from_yaml(get_args())["test"])
