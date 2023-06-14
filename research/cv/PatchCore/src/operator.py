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
"""operator"""
import os
import cv2
import mindspore
import mindspore.ops as ops
import numpy as np


def unfold(img, kernel_size, stride=1, pad=0, dilation=1):
    """
    unfold function
    """
    batch_num, channel, height, width = img.shape
    out_h = (height + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((batch_num, channel, kernel_size, kernel_size, out_h, out_w)).astype(img.dtype)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = np.reshape(col, (batch_num, channel * kernel_size * kernel_size, out_h * out_w))

    return col


def fold(col, input_shape, kernel_size, stride=1, pad=0):
    """
    fold function
    """
    batch_num, channel, height, width = input_shape
    out_h = (height + pad + pad - kernel_size) // stride + 1
    out_w = (width + pad + pad - kernel_size) // stride + 1

    col = col.reshape(batch_num, channel, kernel_size, kernel_size, out_h, out_w)
    img = np.zeros((batch_num, channel, height + pad + pad + stride - 1, width + pad + pad + stride - 1)).astype(
        col.dtype
    )
    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : height + pad, pad : width + pad]


def embedding_concat(x, y):
    """
    embedding_concat function
    """
    B, C1, H1, W1 = x.shape
    _, C2, H2, W2 = y.shape
    s = int(H1 / H2)
    x = unfold(x, s, stride=s)
    x = np.reshape(x, (B, C1, -1, H2, W2))
    z = np.zeros((B, C1 + C2, x.shape[2], H2, W2))

    for i in range(x.shape[2]):
        z[:, :, i, :, :] = np.concatenate((x[:, :, i, :, :], y), axis=1)

    z = np.reshape(z, (B, -1, H2 * W2))
    z = fold(z, (z.shape[0], int(z.shape[1] / (s * s)), H1, W1), s, stride=s)

    return z


def reshape_embedding(embedding):
    """
    reshape_embedding function
    """
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])

    return embedding_list


def prep_dirs(path, category):
    """
    prep_dirs function
    """
    root = os.path.join(path, "checkpoint", category)
    os.makedirs(root, exist_ok=True)

    # make embeddings dir
    embeddings_path = os.path.join(root, "embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    # make sample dir
    sample_path = os.path.join(root, "sample")
    os.makedirs(sample_path, exist_ok=True)

    return embeddings_path, sample_path


def normalize(input_n, mean, std):
    """
    normalize function
    input: numpy
    output: numpy
    """
    mean = mindspore.Tensor(mean, dtype=mindspore.float32).view((-1, 1, 1))
    std = mindspore.Tensor(std, dtype=mindspore.float32).view((-1, 1, 1))

    sub = ops.Sub()
    div = ops.Div()

    out = div(sub(mindspore.Tensor(input_n), mean), std)

    return out.asnumpy()


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
    cv2.imwrite(os.path.join(sample_path, f"{x_type}_{file_name}.jpg"), input_img)
    cv2.imwrite(os.path.join(sample_path, f"{x_type}_{file_name}_amap.jpg"), anomaly_map_norm_hm)
    cv2.imwrite(os.path.join(sample_path, f"{x_type}_{file_name}_amap_on_img.jpg"), hm_on_img)
    cv2.imwrite(os.path.join(sample_path, f"{x_type}_{file_name}_gt.jpg"), gt_img)
