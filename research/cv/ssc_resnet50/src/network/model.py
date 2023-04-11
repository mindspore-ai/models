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
"""Model"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import dtype as mstype
from mindspore import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.common import Tensor

from .grad_clip import GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE, clip_grad
from .resnet import resnet50


class BatchShuffle(nn.Cell):
    """Batch shuffle, for making use of BatchNorm."""

    def __init__(self, args):
        super(BatchShuffle, self).__init__()
        self.rank = args.rank
        self.allgather = ops.AllGather()
        self.concat = ops.Concat(axis=0)
        self.broad = ops.Broadcast(root_rank=0)
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.scalartotensor = ops.ScalarToTensor()
        self.sort = ops.Sort(0)

    def construct(self, x):
        batch_size_this = x.shape[0]
        x_gather = self.allgather(x)
        x_gather = F.stop_gradient(x_gather)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        batch_size_all_tensor = self.scalartotensor(batch_size_all, mstype.int32)
        batch_size_all_tensor = batch_size_all_tensor.reshape((1,))
        idx_shuffle = ops.Randperm(max_length=batch_size_all)(batch_size_all_tensor)

        # broadcast to all gpus
        idx_shuffle_broad = self.broad((idx_shuffle,))

        # index for restoring
        idx_cast = self.cast(idx_shuffle_broad[0], mstype.float32)
        _, idx_unshuffle = self.sort(idx_cast)

        # index for this gpu
        idx_this = self.reshape(idx_shuffle_broad[0], (num_gpus, -1))[self.rank]

        return x_gather[idx_this], idx_unshuffle


class BatchUnShuffle(nn.Cell):
    """Undo batch shuffle."""

    def __init__(self, args):
        super(BatchUnShuffle, self).__init__()
        self.rank = args.rank
        self.allgather = ops.AllGather()
        self.reshape = ops.Reshape()

    def construct(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.allgather(x)
        x_gather = F.stop_gradient(x_gather)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        idx_this = self.reshape(idx_unshuffle, (num_gpus, -1))[self.rank]
        return x_gather[idx_this]


class ModelBaseDis(nn.Cell):
    """Comatch Base Net"""

    def __init__(self, args, is_eval=False):
        super(ModelBaseDis, self).__init__()

        if is_eval:
            self.encode = resnet50(class_num=args.num_clas, mlp=True, low_dim=args.low_dim)
        else:
            self.encode = resnet50(class_num=args.num_clas, mlp=True, low_dim=args.low_dim)
            self.m_encode = resnet50(class_num=args.num_clas, mlp=True, low_dim=args.low_dim)

            self.batch_size = args.batch_size

            # EMA keep rate
            self.m_epoch_zero = Tensor(0.99, mstype.float32)
            self.m_epoch_other = Tensor(0.996, mstype.float32)

            self.rank = args.rank

            # global steps used for record the steps
            self.base_global_steps = Parameter(Tensor(0.0, mstype.float32), name="base_global_steps",
                                               requires_grad=False)
            self.base_global_steps.init_data()
            self.steps_per_epoch = args.steps_per_epoch

            self.weights_encode = ParameterTuple(self.encode.trainable_params())
            self.weights_m_encode = ParameterTuple(self.m_encode.trainable_params())

            for param, param_m in zip(self.encode.trainable_params(), self.m_encode.trainable_params()):
                param_m.set_data(param.data)
                param_m.requires_grad = False

            self.len = len(self.weights_encode)
            self.assign = ops.Assign()
            self.assignadd = ops.AssignAdd()
            self.concat = ops.Concat(axis=0)

            self.batchshuffle = BatchShuffle(args)
            self.batchunshuffle = BatchUnShuffle(args)
            self.select = ops.Select()

    def construct(self, label, unlabel_weak, unlabel_strong0, unlabel_strong1, target, is_eval=False):

        if is_eval:
            outputs_x, _ = self.encode(label)
            return outputs_x, target, _, _

        imgs = self.concat((label, unlabel_strong0))
        outputs, features = self.encode(imgs)  # get the probs and feature which from project head

        # do ema
        # epoch 0 use self.m_epoch_zero, other use self.m_epoch_other
        condition = np.greater(self.base_global_steps, self.steps_per_epoch)
        m = self.select(condition, self.m_epoch_other, self.m_epoch_zero)
        self.assignadd(self.base_global_steps, 1)

        for i in range(self.len):
            self.assign(self.weights_m_encode[i], m * self.weights_m_encode[i])
            self.assignadd(self.weights_m_encode[i], (1 - m) * self.weights_encode[i])

        imgs_m = self.concat((label, unlabel_weak, unlabel_strong1))

        # step1 do batch shuffle
        imgs_m, idx_unshuffle = self.batchshuffle(imgs_m)
        outputs_m, features_m = self.m_encode(imgs_m)

        # step2 do batch unshuffle
        outputs_m = self.batchunshuffle(outputs_m, idx_unshuffle)
        features_m = self.batchunshuffle(features_m, idx_unshuffle)

        return outputs, features, outputs_m, features_m


class ModelWithLossCellDis(nn.Cell):
    """CoMatch loss"""

    def __init__(self, args, network):
        super(ModelWithLossCellDis, self).__init__()
        self.comatch_network = network
        self.K = args.K
        self.K_tensor = Tensor(args.K, mindspore.int32)

        # queue to store momentum feature for strong augmentations
        self.stand_norm = mindspore.common.initializer.Normal(sigma=1, mean=0.0)
        self.queue_s = Parameter(default_input=initializer(self.stand_norm, [args.low_dim, args.K], mstype.float32),
                                 requires_grad=False)
        l2_normalize = ops.L2Normalize(epsilon=1e-12)
        self.queue_s = l2_normalize(self.queue_s)
        self.queue_s.init_data()

        self.queue_s = Parameter(default_input=initializer('Zero', [args.low_dim, args.K], mstype.float32),
                                 requires_grad=False)
        self.queue_s.init_data()

        # queue to store momentum probs for weak augmentations (unlabeled)
        self.probs_u = Parameter(default_input=initializer('Zero', [args.num_clas, args.K], mstype.float32),
                                 requires_grad=False)
        self.probs_u.init_data()

        # point the index of self.queue_s and self.probs_u
        self.queue_s_ptr = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.queue_s_ptr.init_data()

        # queue (memory bank) to store momentum feature and probs for weak augmentations (labeled and unlabeled)
        self.queue_w = Parameter(default_input=initializer(self.stand_norm, [args.low_dim, args.K], mstype.float32),
                                 requires_grad=False)
        self.queue_w.init_data()

        self.probs_xu = Parameter(default_input=initializer('Zero', [args.num_clas, args.K], mstype.float32),
                                  requires_grad=False)
        self.probs_xu.init_data()

        # point the index of self.queue_w and self.probs_xu
        self.queue_w_ptr = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.queue_w_ptr.init_data()

        # used for distribute alignment
        self.hist_prob = Parameter(default_input=initializer('Zero', [args.num_hist, args.num_clas], mstype.float32),
                                   requires_grad=False)
        self.hist_prob.init_data()

        # point the index of self.hist_prob
        self.hist_prob_ptr = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.hist_prob_ptr.init_data()

        self.global_steps = Parameter(Tensor(0.0, mstype.float32), name="global_steps", requires_grad=False)
        self.global_steps.init_data()

        # ops define and instant
        self.softmax = ops.Softmax(axis=1)
        self.mean = ops.ReduceMean()
        self.allreduce = ops.AllReduce()
        self.allgather = ops.AllGather()
        self.assignadd = ops.AssignAdd()
        self.assign = ops.Assign()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.matmul = ops.MatMul()
        self.exp = ops.Exp()
        self.concat = ops.Concat(axis=1)
        self.concat0 = ops.Concat(axis=0)
        self.zeros = ops.Zeros()
        self.reshape = ops.Reshape()
        self.cross_entropy = mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.arg_max_value = ops.ArgMaxWithValue(axis=1)
        self.cast = ops.Cast()
        self.log_softmax = mindspore.nn.LogSoftmax(axis=1)
        self.select = ops.Select()
        self.min = ops.Minimum()

        # init params
        self.low_dim = args.low_dim
        self.num_class = args.num_clas
        self.thr = args.thr
        self.contrast_th = args.contrast_th
        self.lam_u = args.lam_u
        self.lam_c = args.lam_c
        self.num_hist = args.num_hist
        self.alpha = args.alpha
        self.temperature = args.temperature
        self.rank = args.rank
        self.device_num = args.device_num

        # used for label one-hot
        self.depth = args.num_clas

        # label smooth
        self.on_value = Tensor(1.0 - 0.1, mindspore.float32)
        self.off_value = Tensor(1.0 * 0.1 / (args.num_clas - 1), mindspore.float32)
        self.axis = Tensor(0, mindspore.int32)

        # used for control epoch
        self.steps_per_epoch = args.steps_per_epoch

        # used for generate the continuity num
        self.start = Tensor(0, mstype.int32)
        self.limit_low_dim = Tensor(args.low_dim, mstype.int32)
        self.limit_num_class = Tensor(args.num_clas, mstype.int32)
        self.delta = Tensor(1, mstype.int32)

        # avoid using dynamic shapes
        self.zero = Tensor(0, mstype.int32)
        self.limit_low_dim = Tensor(args.low_dim, mstype.int32)
        self.base = ops.range(Tensor(0, mstype.int32),
                              Tensor(args.batch_size * args.unlabel_label * args.device_num, mstype.int32),
                              Tensor(1, mstype.int32))
        self.base_xu = ops.range(Tensor(0, mstype.int32),
                                 Tensor((args.batch_size + args.batch_size * args.unlabel_label) * args.device_num,
                                        mstype.int32),
                                 Tensor(1, mstype.int32))

    def generate_row_index(self, batch_size, dim):
        row_index = ops.Range()(self.zero, dim, self.delta)
        row_index_re = row_index.repeat(batch_size)
        row_index_reshape = row_index_re.reshape(-1, 1)
        return row_index_reshape

    def update_queue_and_prob(self, features, probs, queue_ptr, base, queue, probs_u):
        z = self.allgather(features)
        z = F.stop_gradient(z)
        t = self.allgather(probs)
        t = F.stop_gradient(t)
        batch_size = z.shape[0]

        # step 1.1 generate the row index
        index = queue_ptr % self.K  # self.queue_s_ptr. self.queue_w_ptr
        row_index_reshape = self.generate_row_index(batch_size, self.limit_low_dim)
        row_index_reshape_u = self.generate_row_index(batch_size, self.limit_num_class)

        # step 1 update the self.queue_s or self.queue_w
        limit_now = index + batch_size
        condition = ops.Greater()(limit_now, self.K)
        gap_count = ops.select(condition, self.zero, index)
        btx_index = ops.Add()(base, gap_count)
        btx_index_reshape = btx_index.reshape(-1, 1)
        col_index_re = ops.Tile()(btx_index_reshape, (1, self.low_dim))
        col_flatten = col_index_re.flatten(order='F')
        col_reshape = col_flatten.reshape(-1, 1)
        indices = ops.Concat(axis=-1)((row_index_reshape, col_reshape))
        update_value = z.T.flatten()
        ops.ScatterNdUpdate()(queue, indices, update_value)

        # step 2 update the self.probs_u or self.probs_xu
        col_index_u_re = ops.Tile()(btx_index_reshape, (1, self.num_class))
        col_flatten_u = col_index_u_re.flatten(order='F')
        col_u_reshape = col_flatten_u.reshape(-1, 1)
        indices_u = ops.Concat(axis=-1)((row_index_reshape_u, col_u_reshape))
        update_value_u = t.T.flatten()
        ops.ScatterNdUpdate()(probs_u, indices_u, update_value_u)  # self.probs_u self.probs_xu

        # move pointer
        index_add = ops.select(condition, self.K_tensor - batch_size, index + batch_size) % self.K
        self.assign(queue_ptr, index_add)
        return index_add

    def compute_unsupervised_cross_entropy(self, probs, outputs):
        _, scores = self.arg_max_value(probs)
        scores_com_thr = (scores >= self.thr)
        mask = self.cast(scores_com_thr, mindspore.float32)
        mask = F.stop_gradient(mask)

        # unsupervised cross-entropy
        probs_u_s0 = self.log_softmax(outputs)
        probs_mul = probs_u_s0 * probs
        probs_mul_mask = - mindspore.ops.ReduceSum()(probs_mul, 1) * mask
        loss_u = self.mean(probs_mul_mask, 0)
        return loss_u

    def compute_unsupervised_contrastive_loss(self, them, sim):
        # remove edges with low similairty and normalize pseudo-label graph
        pos_mask = (them >= self.contrast_th)
        pos_mask = self.cast(pos_mask, mindspore.float32)
        them_mask = them * pos_mask

        them_mask = them_mask / self.sum(them_mask, 1)
        positives = sim * pos_mask
        pos_probs = positives / self.sum(sim, 1)
        log_probs = ops.Log()(pos_probs + 1e-7) * pos_mask

        # unsupervised contrastive loss
        log_mul = log_probs * them_mask
        loss_contrast = - ops.ReduceSum()(log_mul, 1)
        loss_contrast = ops.ReduceMean()(loss_contrast, 0)
        return loss_contrast

    def construct(self, label, unlabel_weak, unlabel_strong0, unlabel_strong1, target):
        outputs, features, outputs_m, features_m = self.comatch_network(label, unlabel_weak,
                                                                        unlabel_strong0,
                                                                        unlabel_strong1, target)
        labels_x = target[:, 0]
        btx = label.shape[0]
        btu = unlabel_weak.shape[0]

        outputs_x = outputs[:btx]
        features_u_s0, outputs_u_s0 = features[btx:], outputs[btx:]
        feature_u_w, outputs_u_w = features_m[btx:btx + btu], outputs_m[btx:btx + btu]
        features_u_s1, feature_xu_w = features_m[btx + btu:], features_m[:btx + btu]

        # backpro will cause a lot compute and time consume
        # so we turn off its gradient .deteach()=stop_gradient .clone()
        outputs_u_w = F.stop_gradient(outputs_u_w)
        feature_u_w = F.stop_gradient(feature_u_w)
        feature_xu_w = F.stop_gradient(feature_xu_w)
        features_u_s1 = F.stop_gradient(features_u_s1)

        probs = self.softmax(outputs_u_w)
        probs_bt_avg = self.mean(probs, 0)
        probs_bt_avg_allreduce = self.allreduce(probs_bt_avg)
        probs_bt_avg_allreduce = F.stop_gradient(probs_bt_avg_allreduce)
        probs_bt_avg_dis = probs_bt_avg_allreduce / self.device_num

        index = self.hist_prob_ptr % self.num_hist

        # update the self.hist_prob[index] value
        hist_prob = F.depend(self.hist_prob,
                             ops.ScatterUpdate()(self.hist_prob, index.reshape(1), probs_bt_avg_dis.reshape(1, -1)))

        # increase the self.hist_prob_ptr
        index_add = (self.hist_prob_ptr + 1) % self.num_hist
        self.assign(self.hist_prob_ptr, index_add)

        # DA:distribution alignment
        probs_avg = self.mean(hist_prob, 0)
        probs = probs / probs_avg
        probs_sum = self.sum(probs, 1)
        probs = probs / probs_sum
        probs_orig = F.stop_gradient(probs)

        # memory-smoothed pseudo-label refinement (starting from 2nd epoch)
        matmul_val = self.matmul(feature_u_w, self.queue_w)
        val_dis = matmul_val / self.temperature
        A = self.exp(val_dis)
        A = A / self.sum(A, 1)
        probs_refine = self.alpha * probs + (1 - self.alpha) * self.matmul(A, self.probs_xu.T)

        # use the ops.Select replace the if
        condition = np.greater(self.global_steps, self.steps_per_epoch)
        condition = self.cast(condition, mstype.int32)
        condition = ops.BroadcastTo(probs.shape)(condition)
        condition = self.cast(condition, mstype.bool_)
        probs = self.select(condition, probs_refine, probs)  # can replace if else, if maybe support till mp1.6.0()

        # construct pseudo-label graph
        # similarity with current batch
        Q_self = self.matmul(probs, probs.T)
        Q_self_size = Q_self.shape[0]
        for i in range(Q_self_size):
            Q_self = Q_self.itemset((i, i), 1)

        Q_past = self.matmul(probs, self.probs_u)

        # concatenate them
        Q = self.concat((Q_self, Q_past))

        # construct embedding graph for strong augmentations
        matmul_dis_feature = self.matmul(features_u_s0, features_u_s1.T) / self.temperature
        sim_self = self.exp(matmul_dis_feature)

        matmul_dis_feature_pass = self.matmul(features_u_s0, self.queue_s) / self.temperature
        sim_past = self.exp(matmul_dis_feature_pass)
        sim = self.concat((sim_self, sim_past))

        # update the self.queue_s and self.probs_u use the strong augment unlabel data feature and probs
        self.update_queue_and_prob(features_u_s1, probs, self.queue_s_ptr, self.base, self.queue_s, self.probs_u)

        onehot = ops.OneHot()(target[:, self.axis], self.depth, self.on_value, self.off_value)
        probs_xu = self.concat0((onehot, probs_orig))

        # store label and weak features and probs into memory bank
        self.update_queue_and_prob(feature_xu_w, probs_xu, self.queue_w_ptr, self.base_xu, self.queue_w, self.probs_xu)

        loss_x = self.cross_entropy(outputs_x, labels_x)
        loss_u = self.compute_unsupervised_cross_entropy(probs, outputs_u_s0)
        loss_contrast = self.compute_unsupervised_contrastive_loss(Q, sim)

        ops.assign_add(self.global_steps, 1)
        epoch = self.cast((self.global_steps / self.steps_per_epoch) + 1, mstype.int32)
        lam_c = self.min(self.lam_c, epoch)
        loss = loss_x + self.lam_u * loss_u + lam_c * loss_contrast

        loss_x = F.stop_gradient(loss_x)
        loss_u = F.stop_gradient(loss_u)
        loss_contrast = F.stop_gradient(loss_contrast)

        return loss, loss_x, loss_u, loss_contrast


class TrainOneStepCellDist(nn.Cell):
    """Training wrapper."""

    def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCellDist, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = reduce_flag
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

    def construct(self, label, unlabel_weak, unlabel_strong0, unlabel_strong1, target):
        weights = self.weights
        loss, loss_x, loss_u, loss_contrast = self.network(label, unlabel_weak, unlabel_strong0, unlabel_strong1,
                                                           target)

        sens_1 = P.Fill()(P.DType()(loss), P.Shape()(loss), 1.0)
        sens_2 = P.Fill()(P.DType()(loss_x), P.Shape()(loss_x), 0.0)
        sens_3 = P.Fill()(P.DType()(loss_u), P.Shape()(loss_u), 0.0)
        sens_4 = P.Fill()(P.DType()(loss_contrast), P.Shape()(loss_contrast), 0.0)
        grads = self.grad(self.network, weights)(label, unlabel_weak, unlabel_strong0, unlabel_strong1, target,
                                                 (sens_1, sens_2, sens_3, sens_4))
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss, loss_x, loss_u, loss_contrast


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


@_grad_overflow.register("RowTensor")
def _tensor_grad_overflow_row_tensor(grad):
    return grad_overflow(grad.values)


class TrainOneStepWithLossScaleCellDist(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_update_cell):
        super(TrainOneStepWithLossScaleCellDist, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                self.degree = context.get_auto_parallel_context("device_num")
            else:
                self.degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self, label, unlabel_weak, unlabel_strong0, unlabel_strong1, target, sens=None):
        weights = self.weights
        loss, loss_x, loss_u, loss_contrast = self.network(label, unlabel_weak, unlabel_strong0, unlabel_strong1,
                                                           target)

        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        # sens_1 = P.Fill()(P.DType()(loss), P.Shape()(loss), 1.0)
        sens_2 = P.Fill()(P.DType()(loss_x), P.Shape()(loss_x), 0.0)
        sens_3 = P.Fill()(P.DType()(loss_u), P.Shape()(loss_u), 0.0)
        sens_4 = P.Fill()(P.DType()(loss_contrast), P.Shape()(loss_contrast), 0.0)
        grads = self.grad(self.network, weights)(label, unlabel_weak, unlabel_strong0, unlabel_strong1, target,
                                                 (self.cast(scaling_sens, mstype.float32), sens_2, sens_3, sens_4))

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, loss_x, loss_u, loss_contrast, cond, scaling_sens)
