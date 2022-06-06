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

import numpy as np


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)


def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        overall = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        perclass = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (
            conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)
        ).astype(np.float)
    return overall * 100.0, np.nanmean(perclass) * 100.0, np.nanmean(IU) * 100.0


def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    for m in model.get_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
