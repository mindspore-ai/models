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
"""hub config"""
from src.model_utils.config import net_config
from src import CenterNetMultiPoseLossCell

def centernet_net(*args, **kwargs):
    return CenterNetMultiPoseLossCell(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """create_network about centernet"""
    if name == "centernet":
        net_with_loss = CenterNetMultiPoseLossCell(net_config)
        return net_with_loss
    raise NotImplementedError(f"{name} is not implemented in the repo")
