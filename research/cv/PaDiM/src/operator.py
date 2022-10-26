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
import mindspore.ops as ops
import numpy as np

def unfold(img, kernel_size, stride=1, pad=0, dilation=1):
    """
    unfold function
    """
    batch_num, channel, height, width = img.shape
    out_h = (height + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((batch_num, channel, kernel_size, kernel_size, out_h, out_w)).astype(img.dtype)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = np.reshape(col, (batch_num, channel*kernel_size*kernel_size, out_h*out_w))

    return col

def fold(col, input_shape, kernel_size, stride=1, pad=0):
    """
    fold function
    """
    batch_num, channel, height, width = input_shape
    out_h = (height + pad + pad - kernel_size) // stride + 1
    out_w = (width + pad + pad - kernel_size) // stride + 1

    col = col.reshape(batch_num, channel, kernel_size, kernel_size, out_h, out_w)
    img = np.zeros((batch_num,
                    channel,
                    height + pad + pad + stride - 1,
                    width + pad + pad + stride - 1)) \
        .astype(col.dtype)
    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:height + pad, pad:width + pad]

def embedding_concat(x, y):
    """
    embedding_concat function
    """
    B, C1, H1, W1 = x.shape
    _, C2, H2, W2 = y.shape
    s = int(H1 / H2)
    x = unfold(x, s, stride=s)
    x = np.reshape(x, (B, C1, -1, H2, W2))
    z = np.zeros((B, C1 + C2, x.shape[2], H2, W2), dtype=np.float32)

    for i in range(x.shape[2]):
        z[:, :, i, :, :] = np.concatenate((x[:, :, i, :, :], y), axis=1)

    z = np.reshape(z, (B, -1, H2*W2))
    z = fold(z, (z.shape[0], int(z.shape[1] / (s * s)), H1, W1), s, stride=s)

    return z

def view(embedding_vectors, B, C, H, W):
    split = ops.Split(1, 2)
    embedding_vectors_a, embedding_vectors_b = split(embedding_vectors)
    embedding_vectors_a = embedding_vectors_a.view((B, int(C / 2), H * W))
    embedding_vectors_b = embedding_vectors_b.view((B, int(C / 2), H * W))
    cat = ops.Concat(1)
    embedding_vectors = cat((embedding_vectors_a, embedding_vectors_b))
    return embedding_vectors

def prep_dirs(path, category):
    """
    prep_dirs function
    """
    root = os.path.join(path, category)
    os.makedirs(root, exist_ok=True)

    # make embeddings dir
    embeddings_path = os.path.join(root, 'embeddings')
    os.makedirs(embeddings_path, exist_ok=True)

    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)

    return embeddings_path, sample_path
