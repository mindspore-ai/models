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
"""Generate Dataset"""
import mindspore.dataset as ds


class DataSetNetG():
    """
    Generate Training Dataset to train NetG
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (self.data, self.label)

    def __len__(self):
        return 1


class DataSetNetLoss():
    """
    Generator Training Dataset to train NetF
    """
    def __init__(self, x, nx):
        self.x = x
        self.nx = nx

    def __getitem__(self, index):
        return (self.x, self.nx)

    def __len__(self):
        return 1


def GenerateDataSet(inset, bdset):
    """
    Generator Dataset for training

    Args:
        inset: Inner Set
        bdset: Boundary Set
    """
    datasetnetg = DataSetNetG(bdset.d_x, bdset.d_r)
    datasetnetloss = DataSetNetLoss(inset.x, bdset.n_x)
    DS_NETG = ds.GeneratorDataset(
        datasetnetg, ["data", "label"], shuffle=False)
    DS_NETL = ds.GeneratorDataset(
        datasetnetloss, ["x_inset", "x_bdset"], shuffle=False)
    return DS_NETG, DS_NETL
