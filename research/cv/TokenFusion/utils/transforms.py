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

import cv2
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore as ms

KEYS_TO_DTYPES = {
    "rgb": np.float,
    "depth": np.float,
    "normals": np.float,
    "mask": np.long,
}


class Pad:
    """Pad image and mask to the desired size.

    Args:
      size (int) : minimum length/width.
      img_val (array) : image padding value.
      msk_val (int) : mask padding value.

    """

    def __init__(self, size, img_val, msk_val):
        assert isinstance(size, int)
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image = sample["rgb"]
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        for key in sample["inputs"]:
            sample[key] = self.transform_input(sample[key], pad)
        sample["mask"] = np.pad(
            sample["mask"], pad, mode="constant", constant_values=self.msk_val
        )
        return sample

    def transform_input(self, inp, pad):
        inp = np.stack(
            [
                np.pad(
                    inp[:, :, c],
                    pad,
                    mode="constant",
                    constant_values=self.img_val[c],
                )
                for c in range(3)
            ],
            axis=2,
        )
        return inp


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample["rgb"]
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        for key in sample["inputs"]:
            sample[key] = self.transform_input(sample[key], top, new_h, left, new_w)
        sample["mask"] = sample["mask"][top : top + new_h, left : left + new_w]
        return sample

    def transform_input(self, inp, top, new_h, left, new_w):
        inp = inp[top : top + new_h, left : left + new_w]
        return inp


class ResizeAndScale:
    """Resize shorter/longer side to a given value and randomly scale.

    Args:
        side (int) : shorter / longer side value.
        low_scale (float) : lower scaling bound.
        high_scale (float) : upper scaling bound.
        shorter (bool) : whether to resize shorter / longer side.

    """

    def __init__(self, side, low_scale, high_scale, shorter=True):
        assert isinstance(side, int)
        assert isinstance(low_scale, float)
        assert isinstance(high_scale, float)
        self.side = side
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.shorter = shorter

    def __call__(self, sample):
        image = sample["rgb"]
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if self.shorter:
            min_side = min(image.shape[:2])
            if min_side * scale < self.side:
                scale = self.side * 1.0 / min_side
        else:
            max_side = max(image.shape[:2])
            if max_side * scale > self.side:
                scale = self.side * 1.0 / max_side
        inters = {"rgb": cv2.INTER_CUBIC, "depth": cv2.INTER_NEAREST}
        for key in sample["inputs"]:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], scale, inter)
        sample["mask"] = cv2.resize(
            sample["mask"], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        return sample

    def transform_input(self, inp, scale, inter):
        inp = cv2.resize(inp, None, fx=scale, fy=scale, interpolation=inter)
        return inp


class CropAlignToMask:
    """Crop inputs to the size of the mask."""

    def __call__(self, sample):
        mask_h, mask_w = sample["mask"].shape[:2]
        for key in sample["inputs"]:
            sample[key] = self.transform_input(sample[key], mask_h, mask_w)
        return sample

    def transform_input(self, inp, mask_h, mask_w):
        input_h, input_w = inp.shape[:2]
        if (input_h, input_w) == (mask_h, mask_w):
            return inp
        h, w = (input_h - mask_h) // 2, (input_w - mask_w) // 2
        del_h, del_w = (input_h - mask_h) % 2, (input_w - mask_w) % 2
        inp = inp[h : input_h - h - del_h, w : input_w - w - del_w]
        assert inp.shape[:2] == (mask_h, mask_w)
        return inp


class ResizeAlignToMask:
    """Resize inputs to the size of the mask."""

    def __call__(self, sample):
        mask_h, mask_w = sample["mask"].shape[:2]
        assert mask_h == mask_w
        inters = {"rgb": cv2.INTER_CUBIC, "depth": cv2.INTER_NEAREST}
        for key in sample["inputs"]:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], mask_h, inter)
        return sample

    def transform_input(self, inp, mask_h, inter):
        input_h, input_w = inp.shape[:2]
        assert input_h == input_w
        scale = mask_h / input_h
        inp = cv2.resize(inp, None, fx=scale, fy=scale, interpolation=inter)
        return inp


class ResizeInputs:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if self.size is None:
            return sample
        size = sample["rgb"].shape[0]
        scale = self.size / size
        inters = {"rgb": cv2.INTER_CUBIC, "depth": cv2.INTER_NEAREST}
        for key in sample["inputs"]:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], scale, inter)
        return sample

    def transform_input(self, inp, scale, inter):
        inp = cv2.resize(inp, None, fx=scale, fy=scale, interpolation=inter)
        return inp


class ResizeInputsScale:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        if self.scale is None:
            return sample
        inters = {"rgb": cv2.INTER_CUBIC, "depth": cv2.INTER_NEAREST}
        for key in sample["inputs"]:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], self.scale, inter)
        return sample

    def transform_input(self, inp, scale, inter):
        inp = cv2.resize(inp, None, fx=scale, fy=scale, interpolation=inter)
        return inp


class RandomMirror:
    """Randomly flip the image and the mask"""

    def __call__(self, sample):
        do_mirror = np.random.randint(2)
        if do_mirror:
            for key in sample["inputs"]:
                sample[key] = cv2.flip(sample[key], 1)
            sample["mask"] = cv2.flip(sample["mask"], 1)
        return sample


class Normalise:
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (scale * channel - mean) / std

    Args:
        scale (float): Scaling constant.
        mean (sequence): Sequence of means for R,G,B channels respectively.
        std (sequence): Sequence of standard deviations for R,G,B channels
            respectively.
        depth_scale (float): Depth divisor for depth annotations.

    """

    def __init__(self, scale, mean, std, depth_scale=1.0):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):
        for key in sample["inputs"]:
            if key == "depth":
                continue
            sample[key] = (self.scale * sample[key] - self.mean) / self.std
        if "depth" in sample:
            if self.depth_scale > 0:
                sample["depth"] = self.depth_scale * sample["depth"]
            elif self.depth_scale == -1:  # taskonomy
                sample["depth"] = np.log(1 + sample["depth"])
            elif self.depth_scale == -2:  # sunrgbd
                depth = sample["depth"]
                sample["depth"] = (
                    (depth - depth.min()) * 255.0 / (depth.max() - depth.min())
                )
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key in ["rgb", "depth"]:
            sample[key] = Tensor(sample[key].transpose((2, 0, 1)), ms.float32)
        sample["mask"] = Tensor(sample["mask"], ms.int64)
        return sample


class ToBatchTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key in ["rgb", "depth"]:
            sample[key] = ops.Transpose()(Tensor(sample[key], ms.float32), (0, 3, 1, 2))
        sample["mask"] = Tensor(sample["mask"], ms.int64)
        return sample


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]
