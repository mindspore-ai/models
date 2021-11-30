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

"""return generator"""

from src.model.ddbpn import Net as DDBPN
from src.model.dbpn import Net as DBPN
from src.model.dbpns import Net as DBPNS
from src.model.ddbpnl import Net as DDBPNL
from src.model.dbpn_iterative import Net as DBPN_ITERATIVE
from src.util.utils import init_weights

def get_generator(model_type, scale_factor=4):
    """return generator network"""
    model = DDBPN(3, 64, 256, 7, scale_factor)
    if model_type == 'DDBPN':
        model = DDBPN(3, 64, 256, 7, scale_factor)
    elif model_type == 'DBPN':
        model = DBPN(3, 64, 256, 10, scale_factor)
    elif model_type == 'DBPNS':
        model = DBPNS(3, 32, 128, 2, scale_factor)
    elif model_type == 'DDBPNL':
        model = DDBPNL(3, 32, 128, 6, scale_factor)
    elif model_type == 'DBPN_ITERATIVE':
        model = DBPN_ITERATIVE(3, 64, 256, 7, scale_factor)
    init_weights(model, 'KaimingNormal', 0.02)
    return model
