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

"""model"""

import src.networks as networks
from src.networks import init_weights
from mindspore import ops
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def get_model(args):
    """
        This class implements the arbitrary style transfer model.
    """
    net = style_transfer_model(args, style_dim=args.style_dim)
    init_weights(net, args.init_type, args.init_gain)
    return net


class style_transfer_model(nn.Cell):
    """
        style_transfer_model.
        Args:
            style_dim (int): The dimension of style vector. Default: 100.
        Returns:
            Tensor, stylied image.
    """

    def __init__(self, args, style_dim=100):
        super(style_transfer_model, self).__init__()
        self.style_prediction_network = networks.style_prediction_network(args, style_dim=style_dim)
        self.style_transfer_network = networks.style_transfer_network(args, style_dim=style_dim)

    def construct(self, content_img, style_feat):
        """construct"""
        style_vector = self.style_prediction_network(style_feat)
        stylied_img = self.style_transfer_network(content_img, style_vector)
        return stylied_img

    def construct_interpolation(self, content_img, style_feat_1, style_feat_2):
        """construct_interpolation"""
        style_vector_1 = self.style_prediction_network(style_feat_1)
        style_vector_2 = self.style_prediction_network(style_feat_2)
        stylied_img_1 = self.style_transfer_network(content_img, style_vector_1 * 1.0 + style_vector_2 * 0.0)
        stylied_img_2 = self.style_transfer_network(content_img, style_vector_1 * 0.8 + style_vector_2 * 0.2)
        stylied_img_3 = self.style_transfer_network(content_img, style_vector_1 * 0.6 + style_vector_2 * 0.4)
        stylied_img_4 = self.style_transfer_network(content_img, style_vector_1 * 0.4 + style_vector_2 * 0.6)
        stylied_img_5 = self.style_transfer_network(content_img, style_vector_1 * 0.2 + style_vector_2 * 0.8)
        stylied_img_6 = self.style_transfer_network(content_img, style_vector_1 * 0.0 + style_vector_2 * 1.0)
        out = [stylied_img_1, stylied_img_2, stylied_img_3, stylied_img_4, stylied_img_5, stylied_img_6]
        return out


class TrainOnestepStyleTransfer(nn.Cell):
    """
    Encapsulation class of arbitrary style transfer network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    """

    def __init__(self, G, optimizer, sens=1.0):
        super(TrainOnestepStyleTransfer, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.G.vgg_conv1.set_grad(False)
        self.G.vgg_conv1.set_train(False)
        self.G.vgg_conv2.set_grad(False)
        self.G.vgg_conv2.set_train(False)
        self.G.vgg_conv3.set_grad(False)
        self.G.vgg_conv3.set_train(False)
        self.G.vgg_conv4.set_grad(False)
        self.G.vgg_conv4.set_train(False)
        self.G.inception.set_grad(False)
        self.G.inception.set_train(False)
        self.G.meanshift.set_grad(False)
        self.G.meanshift.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, HR_img, LR_img):
        """construct"""
        weights = self.weights
        lg = self.G(HR_img, LR_img)
        sens_g = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.G, weights)(HR_img, LR_img, sens_g)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)
        return ops.depend(lg, self.optimizer(grads_g))
