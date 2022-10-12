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
"""Loss function for the FastFlow Model Implementation."""

import mindspore.nn as nn
import mindspore.ops as ops

class FastflowLoss(nn.Cell):
    """Fastflow loss."""
    def __init__(self):
        super(FastflowLoss, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sum = ops.ReduceSum(keep_dims=False)

    def construct(self, hidden_variables, jacobians):
        """Calculate the Fastflow loss.

        Args:
            hidden_variables (List[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (List[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: Fastflow loss computed based on the hidden variables and the log of the Jacobians.
        """
        loss = 0
        for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
            loss += self.mean(0.5 * self.sum(hidden_variable**2, (1, 2, 3)) - jacobian)
        return loss

class NetWithLossCell(nn.Cell):
    '''NetWithLossCell'''
    def __init__(self, network, loss):
        super(NetWithLossCell, self).__init__(auto_prefix=True)
        self.network = network
        self.fastflow_loss = loss

    def construct(self, images):
        hidden_variable, jacobian = self.network(images)
        loss = self.fastflow_loss(hidden_variable, jacobian)
        return loss
