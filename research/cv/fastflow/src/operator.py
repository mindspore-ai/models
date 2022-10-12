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
"""operator"""
import os
import cv2
import numpy as np

def prep_dirs(path, category):
    """
    prep_dirs function
    """
    root = os.path.join(path, 'visible_imgs', category)
    os.makedirs(root, exist_ok=True)

    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)

    return root, sample_path

def normalize(inputs, mean, std):
    """
    normalize function
    """
    mean = np.array(mean).reshape((-1, 1, 1))
    std = np.array(std).reshape((-1, 1, 1))

    out = np.divide(np.subtract(inputs, mean), std).astype(np.float32)

    return out

def cvt2heatmap(gray):
    """
    cvt2heatmap function
    """
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def min_max_norm(image):
    """
    min_max_norm function
    """
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def heatmap_on_image(heatmap, image):
    """
    heatmap_on_image function
    """
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)

def save_anomaly_map(sample_path, anomaly_map, input_img, gt_img, file_name, x_type):
    """
    save_anomaly_map function
    """
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_img)

    # save images
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}.jpg'), input_img)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)
