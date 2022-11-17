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
import os

from src.utils.get_action_history_onehot import get_action_history_onehot
from src.models.vggm import vggm

from mindspore import dtype as mstype
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from mindspore.common.initializer import initializer
from mindspore import numpy as nps
from mindspore.train.serialization import load_checkpoint, load_param_into_net

pretrained_settings = {
    'adnet': {
        'input_space': 'BGR',
        'input_size': [3, 112, 112],
        'input_range': [0, 255],
        'mean': [123.68, 116.779, 103.939],
        'std': [1, 1, 1],
        'num_classes': 11
    }
}


class ADNetDomainSpecific(nn.Cell):
    """
    This module purpose is only for saving the state_dict's domain-specific layers of each domain.
    Put this module to CPU
    """
    def __init__(self, num_classes, num_history):
        super(ADNetDomainSpecific, self).__init__()
        action_dynamic_size = num_classes * num_history
        self.fc6 = nn.Dense(512 + action_dynamic_size, num_classes)
        self.fc7 = nn.Dense(512 + action_dynamic_size, 2)

    def load_weights(self, base_file, video_index, run_online=False):
        """
        Load weights from file
        Args:
            base_file: (string)
            video_index: (int)
        """
        #/cache/weight/ADNet_SL_epoch29.ckpt
        other, ext = os.path.splitext(base_file)
        if ext == '.ckpt':
            print('Loading ADNetDomainSpecific ' + str(video_index) + ' weights')

            if len(other.split('_')) > 3:
                filename_ = other.split('_')[2] + '_' + other.split('_')[3]
            else:
                filename_ = other.split('_')[2]
            if run_online == 'True':
                filename_ = os.path.join('/cache/weight', 'domain_weights', filename_ + '_')
            else:
                filename_ = os.path.join('weights', 'domain_weights', filename_ + '_')
            checkpoint = load_checkpoint(filename_ + str(video_index) + '.ckpt')
            load_param_into_net(self, checkpoint)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_weights_from_adnet(self, adnet_net):
        """
        Load weights from ADNet. Use it after updating adnet to update the weights in this module
        Args:
            adnet_net: (ADNet) the updated ADNet whose fc6 and fc7
        """
        # parameters_dict()
        adnet_state_dict = adnet_net.parameters_dict()
        model_dict = self.parameters_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in adnet_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        for key in pretrained_dict:
            update = nn.ParameterUpdate(model_dict[key])
            update.phase = "update_param"
            update(Tensor(pretrained_dict[key]))


class ADNet(nn.Cell):

    def __init__(self, base_network, opts, num_classes=11, phase='train', num_history=10):
        super(ADNet, self).__init__()

        self.num_classes = num_classes
        self.phase = phase
        self.opts = opts

        self.base_network = base_network
        self.fc4_5 = nn.SequentialCell([
            nn.Dense(18432, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(512, 512),  # [3]
            nn.ReLU(),
            nn.Dropout(0.5)
        ])

        # -1 to differentiate between action '0' and haven't been explored
        self.action_history = Parameter(nps.full((num_history,), -1))

        self.action_dynamic_size = num_classes * num_history
        self.action_dynamic = Parameter(nps.zeros((self.action_dynamic_size,)))

        self.fc6 = nn.Dense(512 + self.action_dynamic_size, self.num_classes)
        self.fc7 = nn.Dense(512 + self.action_dynamic_size, 2)
        self.ops_concat = ops.Concat(1)
        self.ops_softmax = ops.Softmax()
        self.expand = ops.ExpandDims()
    # update_action_dynamic: history of action. We don't update the action_dynamic in SL learning.
    def construct(self, x, action_d=None, update_action_dynamic=False):
        """
        Args:
            x: (Tensor) the input of network
            action_dynamic: (Tensor) the previous state action dynamic.
                If None, use the self.action_dynamic in this Module
            update_action_dynamic: (bool) Whether to update the action_dynamic with the result.
                We don't update the action_dynamic in SL learning.
        """
        x = self.base_network(x)
        x = x.view(x.shape[0], -1)
        x = self.fc4_5(x)

        if action_d is None or action_d == -1:
            ac_d = ops.ExpandDims()(self.action_dynamic, 0)
            ac_d = nps.tile(ac_d, (x.shape[0], 1))
            x = self.ops_concat((x, ac_d))
        else:
            x = self.ops_concat((x, action_d))
        fc6_out = self.fc6(x)
        fc7_out = self.fc7(x)

        if self.phase == 'test':
            fc6_out = self.ops_softmax(fc6_out)
            fc7_out = self.ops_softmax(fc7_out)

        if update_action_dynamic:
            selected_action = ops.Argmax(1, mstype.int32)(fc6_out)
            self.action_history[1:] = self.action_history[0:-1]
            self.action_history[0] = selected_action

        return fc6_out, fc7_out

    def load_domain_specific(self, adnet_domain_specific):
        """
        Load existing domain_specific weight to this model (i.e. fc6 and fc7). Do it before updating this model to
        update the weight to the specific domain
        Args:
             adnet_domain_specific: (ADNetDomainSpecific) the domain's ADNetDomainSpecific module.
        """
        domain_specific_state_dict = adnet_domain_specific.parameters_dict()
        model_dict = self.parameters_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in domain_specific_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        for key in pretrained_dict:
            update = nn.ParameterUpdate(model_dict[key])
            update.phase = "update_param"
            update(Tensor(pretrained_dict[key].asnumpy()))


    def load_weights(self, base_file, load_domain_specific=None):
        """
        Args:
            base_file: (string) checkpoint filename
            load_domain_specific: (None or int) None if not loading.
                Fill it with int of the video idx to load the specific domain weight
        """
        _, ext = os.path.splitext(base_file)
        if ext == '.ckpt':
            print('Loading weights into state dict...')

            pretrained_dict = load_checkpoint(base_file)

            # load adnet

            model_dict = self.parameters_dict()

            # create new OrderedDict that does not contain `module.`

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            for key in pretrained_dict:
                update = nn.ParameterUpdate(model_dict[key])
                update.phase = "update_param"
                update(Tensor(pretrained_dict[key].asnumpy()))

            print('Finished!')
        else:
            print('Sorry only .ckpt files supported.')

    def update_action_dynamic(self, action_history):
        onehot_action = get_action_history_onehot(action_history, self.opts)

        self.action_dynamic.set_data(onehot_action)
        return True

    def reset_action_dynamic(self):
        self.action_dynamic.set_data(nps.zeros((self.action_dynamic_size,)))
        return True

    def get_action_dynamic(self):
        return self.action_dynamic

    def set_phase(self, phase):
        self.phase = phase


def adnet(opts, base_network='vggm', trained_file=None, random_initialize_domain_specific=False,
          multidomain=True, distributed=False, run_online='False'):
    """
    Args:
        base_network: (string)
        trained_file: (None or string) saved filename
        random_initialize_domain_specific: (bool) if there is trained file, whether to use the weight in the file (True)
            or just random initialize (False). Won't matter if the trained_file is None (always False)
        multidomain: (bool) whether to have separate weight for each video or not. Default True: separate
    Returns:
        adnet_model: (ADNet)
        domain_nets: (list of ADNetDomainSpecific) length: #videos
    """
    assert base_network in ['vggm'], "Base network variant is unavailable"

    num_classes = opts['num_actions']
    num_history = opts['num_action_history']

    assert num_classes in [11], "num classes is not exist"

    settings = pretrained_settings['adnet']

    if base_network == 'vggm':
        base_network = vggm()  # by default, load vggm's weights too
        base_network = base_network.features[0:10]

    else:  # change this part if adding more base network variant
        base_network = vggm()
        base_network = base_network.features[0:10]

    if trained_file:
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        print('Resuming training, loading {}...'.format(trained_file))

        adnet_model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes, num_history=num_history)

        adnet_model.load_weights(trained_file)

        adnet_model.input_space = settings['input_space']
        adnet_model.input_size = settings['input_size']
        adnet_model.input_range = settings['input_range']
        adnet_model.mean = settings['mean']
        adnet_model.std = settings['std']
    else:
        adnet_model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes)

    # initialize domain-specific network
    domain_nets = []
    if multidomain:
        num_videos = opts['num_videos']
    else:
        num_videos = 1

    for idx in range(num_videos):
        domain_nets.append(ADNetDomainSpecific(num_classes=num_classes, num_history=num_history))

        scal = Tensor([0.01], mstype.float32)

        if trained_file and not random_initialize_domain_specific:
            domain_nets[idx].load_weights(trained_file, idx, run_online)
        else:
            if distributed:
                domain_nets[idx].init_parameters_data(auto_parallel_mode=True)
            else:
                domain_nets[idx].init_parameters_data(auto_parallel_mode=False)
            # fc 6
            domain_nets[idx].fc6.weight.set_data(
                initializer('Normal', domain_nets[idx].fc6.weight.shape, mstype.float32))
            domain_nets[idx].fc6.weight.set_data(
                domain_nets[idx].fc6.weight.data * scal.expand_as(domain_nets[idx].fc6.weight.data))
            domain_nets[idx].fc6.bias.set_data(nps.full(shape=domain_nets[idx].fc6.bias.shape, fill_value=0.))
            # fc 7
            domain_nets[idx].fc7.weight.set_data(
                initializer('Normal', domain_nets[idx].fc7.weight.shape, mstype.float32))
            domain_nets[idx].fc7.weight.set_data(
                domain_nets[idx].fc7.weight.data * scal.expand_as(domain_nets[idx].fc7.weight.data))
            domain_nets[idx].fc7.bias.set_data(nps.full(shape=domain_nets[idx].fc7.bias.shape, fill_value=0.))

    return adnet_model, domain_nets

class WithLossCell_ADNET(nn.Cell):
    r"""
    Cell with loss function.

    Wraps the network with loss function. This Cell accepts data and label as inputs and
    the computed loss will be returned.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If dtype of `data` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, backbone, loss_fn, phase):
        super(WithLossCell_ADNET, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.phase = phase

    def construct(self, data, label):
        fc6_out, fc7_out = self._backbone(data)
        if self.phase == 'score':
            return self._loss_fn(fc7_out, label)
        return self._loss_fn(fc6_out, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone


class SoftmaxCrossEntropyExpand(nn.Cell):
    '''
        used to train in distributed training
    '''
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_tensor(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
