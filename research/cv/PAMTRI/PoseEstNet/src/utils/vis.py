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
"""vis"""
import os
import math
import cv2
import numpy as np

from .inference import get_max_preds
from .grid import make_grid, permute

def is_convex(kpts):
    """is_convex"""
    dx0 = kpts[-1][0] - kpts[-2][0]
    dy0 = kpts[-1][1] - kpts[-2][1]
    dx1 = kpts[0][0] - kpts[-1][0]
    dy1 = kpts[0][1] - kpts[-1][1]
    x_prod = dx0 * dy1 - dy0 * dx1
    sign_init = np.sign(x_prod)
    if sign_init == 0:
        return False

    for k in range(1, len(kpts)):
        dx0 = kpts[k-1][0] - kpts[k-2][0]
        dy0 = kpts[k-1][1] - kpts[k-2][1]
        dx1 = kpts[k][0] - kpts[k-1][0]
        dy1 = kpts[k][1] - kpts[k-1][1]
        x_prod = dx0 * dy1 - dy0 * dx1
        if sign_init != np.sign(x_prod):
            return False

    return True

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    batch_image = batch_image.asnumpy()
    grid = make_grid(batch_image)
    ndarr = permute(grid)
    ndarr = ndarr.copy()

    nmaps = batch_image.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.shape[2] + padding)
    width = int(batch_image.shape[3] + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    batch_image = batch_image.asnumpy()
    batch_heatmaps = batch_heatmaps.asnumpy()
    if normalize:
        _min = float(np.min(batch_image))
        _max = float(np.max(batch_image))
        batch_image = batch_image - _min
        batch_image = batch_image / (_max - _min + 1e-5)

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros((batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3), dtype=np.uint8)

    preds, _ = get_max_preds(batch_heatmaps)

    for i in range(batch_size):
        image = permute(batch_image[i])
        heatmaps = batch_heatmaps[i] * 255
        heatmaps = np.clip(heatmaps, a_min=0, a_max=255)
        heatmaps = heatmaps.astype(np.uint8)

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, _input, joints, joints_vis, target, joints_pred, output,
                      prefix):
    """save_debug_images"""
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            _input, joints, joints_vis,
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            _input, joints_pred, joints_vis,
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            _input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            _input, output, '{}_hm_pred.jpg'.format(prefix)
        )

def save_heatmaps(batch_heatmaps, batch_image_names, output_heapmap_dir):
    """save_heatmaps"""
    batch_heatmaps = batch_heatmaps.asnumpy()
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]

    for i in range(batch_size):
        heatmaps = batch_heatmaps[i]
        heatmaps = heatmaps * 255
        heatmaps = np.clip(heatmaps, a_min=0, a_max=255)
        heatmaps = heatmaps.astype(np.uint8)

        output_sub_dir = os.path.join(output_heapmap_dir, batch_image_names[i][:-4])

        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)

        for j in range(num_joints):
            file_name = os.path.join(output_sub_dir,
                                     "%02d.jpg" % j)
            cv2.imwrite(file_name, heatmaps[j, :, :])
