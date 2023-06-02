# Copyright 2023 Huawei Technologies Co., Ltd
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
from PIL import Image
import mindspore as ms
from mindspore import ops


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    onsets = (1 - (onsets.t() > onset_threshold).to(ms.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(ms.uint8)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = ops.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)
