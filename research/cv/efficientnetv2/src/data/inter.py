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
"""Interpolation for MindSpore"""
from mindspore.dataset.vision import Inter

Interpolation = {
    "bicubic": Inter.BICUBIC,
    "cubic": Inter.CUBIC,
    "nearest": Inter.NEAREST,
    "area": Inter.AREA,
    "pilcubic": Inter.PILCUBIC,
    "bilinear": Inter.BILINEAR,
    "linear": Inter.LINEAR,
    "antialias": Inter.ANTIALIAS
}
