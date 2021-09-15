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
from src.ncf import NCFModel
from src.config import cfg

def ncf_net(*args, **kwargs):
    return NCFModel(*args, **kwargs)


def create_network(name, *args, **kwargs):
    """create_network about ncf"""
    if name == "ncf":
        layers = cfg.layers
        num_factors = cfg.num_factors
        num_users = 6040
        num_items = 3706
        return ncf_net(num_users=num_users,
                       num_items=num_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)
    raise NotImplementedError(f"{name} is not implemented in the repo")
