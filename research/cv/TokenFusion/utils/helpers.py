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

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

IMG_SCALE = 1.0 / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
logger = None


def print_log(message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + "\n")


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def make_validation_img(img_, depth_, lab, pre):
    # download cmap.npy from https://github.com/yikaiw/CEN/tree/master/semantic_segmentation/utils # noqa
    cmap = np.load("./utils/cmap.npy")

    img = np.array(
        [i * IMG_STD.reshape((3, 1, 1)) + IMG_MEAN.reshape((3, 1, 1)) for i in img_]
    )
    img *= 255
    img = img.astype(np.uint8)
    img = np.concatenate(img, axis=1)

    depth_ = depth_[0].transpose(1, 2, 0) / max(depth_.max(), 10)
    vmax = np.percentile(depth_, 95)
    normalizer = mpl.colors.Normalize(vmin=depth_.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    depth = (mapper.to_rgba(depth_)[:, :, :3] * 255).astype(np.uint8)
    lab = np.concatenate(lab)
    print(lab.shape)
    lab = np.array([cmap[i.astype(np.uint8) + 1] for i in lab])

    pre = np.concatenate(pre)
    pre = np.array([cmap[i.astype(np.uint8) + 1] for i in pre])
    img = img.transpose(1, 2, 0)

    return np.concatenate([img, depth, lab, pre], 1)
