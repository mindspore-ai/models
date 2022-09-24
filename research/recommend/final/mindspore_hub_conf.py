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
"""hub config."""
from src.final import ModelBuilder
from src.model_utils.config import config

def create_network(name, *args, **kwargs):
    if name == 'final':
        model_builder = ModelBuilder(config, config)
        _, final_eval_net = model_builder.get_train_eval_net()
        return final_eval_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
