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
from mindspore import Tensor, ops
import mindspore as ms


def predict_pnet(data, net):
    """Predict data by PNet"""
    data = Tensor(data, dtype=ms.float32)[None, :]
    cls_prob, box_pred, _ = net(data)
    softmax = ops.Softmax(axis=0)
    cls_prob = softmax(cls_prob)

    return cls_prob.asnumpy(), box_pred.asnumpy()

def predict_rnet(data, net):
    """Predict data by RNet"""
    data = Tensor(data, dtype=ms.float32)
    cls_prob, box_pred, _ = net(data)
    softmax = ops.Softmax()
    cls_prob = softmax(cls_prob)
    return cls_prob.asnumpy(), box_pred.asnumpy()

def predict_onet(data, net):
    """Predict data by ONet"""
    data = Tensor(data, dtype=ms.float32)
    cls_prob, box_pred, landmark_pred = net(data)
    softmax = ops.Softmax()
    cls_prob = softmax(cls_prob)
    return cls_prob.asnumpy(), box_pred.asnumpy(), landmark_pred.asnumpy()
