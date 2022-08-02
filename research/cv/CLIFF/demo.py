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

import argparse
import cv2
import mindspore
from mindspore import Tensor
import numpy as np

from models.cliff_res50 import MindSporeModel as cliff_res50
from common.imutils import process_image
from common import constants


def main(args):
    # load the model
    print("ckpt:", args.ckpt)
    cliff = cliff_res50()
    param_dict = mindspore.load_checkpoint(args.ckpt)
    mindspore.load_param_into_net(cliff, param_dict)

    # load and pre-process the image
    print("input_path:", args.input_path)
    img_bgr = cv2.imread(args.input_path)
    img_rgb = img_bgr[:, :, ::-1]
    norm_img, center, scale = process_image(img_rgb, bbox=None)
    norm_img = norm_img[np.newaxis, :, :, :]

    # calculate the bbox info
    cx, cy, b = center[0], center[1], scale * 200
    img_h, img_w, _ = img_rgb.shape
    focal_length = (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree
    bbox_info = np.array([cx - img_w / 2., cy - img_h / 2., b], dtype=np.float32)
    bbox_info = bbox_info[np.newaxis, :]
    bbox_info[:, :2] = bbox_info[:, :2] / focal_length * 2.8  # [-1, 1]
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

    # load the initial parameter
    mean_params = np.load(constants.SMPL_MEAN_PARAMS)
    init_pose = mean_params['pose'][np.newaxis, :].astype('float32')
    init_shape = mean_params['shape'][np.newaxis, :].astype('float32')
    init_cam = mean_params['cam'][np.newaxis, :].astype('float32')

    # feed-forward
    pred_rotmat_6d, pred_betas, pred_cam_crop = cliff(Tensor(norm_img), Tensor(bbox_info),
                                                      Tensor(init_pose), Tensor(init_shape), Tensor(init_cam))
    print("pred_rotmat_6d", pred_rotmat_6d)
    print("pred_betas", pred_betas)
    print("pred_cam_crop", pred_cam_crop)
    print("Inference finished successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='data/im07937.png', help='path to the input data')
    parser.add_argument('--ckpt', default="ckpt/cliff-res50-PA45.7_MJE72.0_MVE85.3_3dpw.ckpt",
                        help='path to the pretrained checkpoint')

    arguments = parser.parse_args()
    main(arguments)
