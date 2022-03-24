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
""" dataset factory """

from .cuhk03 import CUHK03
from .duke import Duke
from .market import Market

__factory = {
    'market': Market,
    'duke': Duke,
    'cuhk03': CUHK03
}


def names():
    """ Get factory names """
    return sorted(__factory.keys())


def create(dataset_name, root, subset_name, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    dataset_name : str
        The dataset name. Can be one of 'market', 'duke', 'cuhk03'
    root : str
        The path to the dataset directory.
    subset_name : str
        The subset name. Can be one of "train", "query", "gallery"
    """
    if dataset_name not in __factory:
        raise KeyError("Unknown dataset:", dataset_name)
    return __factory[dataset_name](root, subset_name, *args, **kwargs)
