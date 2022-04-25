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
""" Dataset factory """

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
