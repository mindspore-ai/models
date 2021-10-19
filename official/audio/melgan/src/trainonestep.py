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
"""TrainOneStepCell definition"""
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.context  import ParallelMode
from mindspore.nn.cell import Cell
from mindspore.common.parameter import ParameterTuple
from mindspore.ops.operations import NPUGetFloatStatus, NPUAllocFloatStatus, NPUClearFloatStatus, ReduceSum, \
    LessEqual
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import composite as C

class TrainOneStepCellDIS(Cell):
    """TrainOneStepCell Discriminator"""
    def __init__(self, networkg, network, optimizer, criterion_d, scale_update_cell=None):
        super(TrainOneStepCellDIS, self).__init__(auto_prefix=False)
        self.networkg = networkg
        self.network = network
        self.criterion = criterion_d
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.less_equal = LessEqual()
        self.allreduce = P.AllReduce()
        self.hyper_map = C.HyperMap()
        self.base = Tensor(1, mstype.float32)
        self.alloc_status = NPUAllocFloatStatus()
        self.get_status = NPUGetFloatStatus()
        self.clear_status = NPUClearFloatStatus()
        self.reduce_sum = ReduceSum(keep_dims=False)
        self.is_distributed = parallel_mode != ParallelMode.STAND_ALONE
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell

    def construct(self, mel, wav, sens):
        """backward network"""
        gdata = self.networkg(mel)
        loss = self.network(gdata, wav)
        # init overflow buffer
        init = self.alloc_status()
        # clear overflow buffer
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(sens, clear_status)

        grads = self.grad(self.network, self.weights)(gdata, wav, F.cast(scaling_sens, F.dtype(loss)))
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        # get the overflow buffer
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)

        # sum overflow buffer elements, 0:not overflow , >0:overflow
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        return F.depend(loss, self.optimizer(grads)), loss, cond


class TrainOneStepCellGEN(Cell):
    """TrainOneStepCell Generator"""
    def __init__(self, network, optimizer, postnetwork, criterion_g, scale_update_cell=None):
        super(TrainOneStepCellGEN, self).__init__(auto_prefix=False)
        self.network = network
        self.postnetwork = postnetwork
        self.criterion = criterion_g
        self.weights = ParameterTuple(network.trainable_params())
        self.postweights = ParameterTuple(postnetwork.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.postgrad = C.GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.cast = P.Cast()
        self.is_distributed = parallel_mode != ParallelMode.STAND_ALONE
        self.less_equal = LessEqual()
        self.base = Tensor(1, mstype.float32)
        self.allreduce = P.AllReduce()
        self.alloc_status = NPUAllocFloatStatus()
        self.get_status = NPUGetFloatStatus()
        self.clear_status = NPUClearFloatStatus()
        self.reduce_sum = ReduceSum(keep_dims=False)
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell

    def construct(self, wav, mel, sens):
        """backward network"""
        gdata = self.network(mel)
        loss = self.postnetwork(gdata, wav)
        # init overflow buffer
        init = self.alloc_status()
        # clear overflow buffer
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(sens, clear_status)

        grads_d = self.postgrad(self.postnetwork, self.postweights)(gdata, wav, F.cast(scaling_sens, F.dtype(loss)))
        sens_g = grads_d[0][0]
        grads_g = self.grad(self.network, self.weights)(mel, sens_g)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)
        # get the overflow buffer
        init = F.depend(init, grads_g)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)

        # sum overflow buffer elements, 0:not overflow , >0:overflow
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        return F.depend(loss, self.optimizer(grads_g)), loss, cond
