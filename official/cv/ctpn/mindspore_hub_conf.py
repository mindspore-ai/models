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
from src.ctpn import CTPN
from src.model_utils.config import config

def create_network(name, *args, **kwargs):
    """create_network about ctpn"""
    if name == "ctpn":
        config.feature_shapes = [config.img_height // 16, config.img_width // 16]
        config.num_bboxes = (config.img_height // 16) * (config.img_width // 16) * config.num_anchors
        config.num_step = config.img_width // 16
        config.rnn_batch_size = config.img_height // 16
        net = CTPN(config=config, batch_size=config.batch_size)
        return net
    raise NotImplementedError(f"{name} is not implemented in the repo")
