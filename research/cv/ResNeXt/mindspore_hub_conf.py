# Copyright 2020 Huawei Technologies Co., Ltd
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
"""hub config."""
from src.image_classification import get_network


def create_network(name, *args, **kwargs):
    if name == "resnext50":
        return get_network("resnext50", *args, **kwargs)
    if name == "resnext101":
        return get_network("resnext101", *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
