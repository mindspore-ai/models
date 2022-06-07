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
"""init model"""
from .vig import vig_ti_patch16_224
from .vig import vig_s_patch16_224
from .vig import vig_b_patch16_224

__all__ = [
    "vig_ti_patch16_224",
    "vig_s_patch16_224",
    "vig_b_patch16_224",
]
