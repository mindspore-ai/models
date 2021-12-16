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
from src.edcn import ModelBuilder
from src.model_utils.config import model_config, train_config

def create_network(name, *args, **kwargs):
    if name == 'edcn':
        model_builder = ModelBuilder(model_config, train_config)
        _, edcn_eval_net = model_builder.get_train_eval_net()
        return edcn_eval_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
