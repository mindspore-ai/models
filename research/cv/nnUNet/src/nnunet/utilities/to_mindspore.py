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

"""covert numpy to mindspore tensor"""

import mindspore


def maybe_to_mindspore(d):
    """maybe to mindspore"""
    if isinstance(d, list):
        d = [maybe_to_mindspore(i) if not isinstance(i, mindspore.Tensor) else i for i in d]
    elif not isinstance(d, mindspore.Tensor):
        d = mindspore.Tensor(d, dtype=mindspore.float32)
    return d
