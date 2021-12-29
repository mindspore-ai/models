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
"""
utils for lstm-crf.
"""
import math
import numpy as np

from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
from mindspore.nn.metrics import ConfusionMatrixMetric
from mindspore.train.callback import Callback
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from .LSTM_CRF import postprocess


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
NONE = "O"


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class Lstm_CRF_Cell_CPU(nn.Cell):
    """LSTM_CRF model"""
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(Lstm_CRF_Cell_CPU, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.grad_reducer = None
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

        self.base = Tensor(1, mstype.float32)
        self.float_status = P.FloatStatus()
        self.addn = P.AddN()
        self.reshape = P.Reshape()
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.cast = P.Cast()
        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self, features, label):
        """LSTM-CRF Finetune cpu"""
        weights = self.weights
        loss = self.network(features, label)
        scaling_sens = self.loss_scale
        grads = self.grad(self.network, weights)(features,
                                                 label,
                                                 self.cast(scaling_sens, mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        self.optimizer(grads)
        return loss


class Lstm_CRF_Cell_Ascend(nn.Cell):
    """add gradient to net"""
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(Lstm_CRF_Cell_Ascend, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.allreduce = P.AllReduce()
        self.grad_reducer = None
        self.cast = P.Cast()
        self.gpu_target = False
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  features,
                  labels,
                  sens=None):
        """LSTM-CRF Finetune"""

        weights = self.weights
        init = False
        loss = self.network(features,
                            labels)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        init = self.alloc_status()
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(features,
                                                 labels,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1, target_device='CPU'):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.target_device = target_device

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            if self.target_device == 'CPU':
                print("epoch: {}, current epoch percent: {}, step: {}, loss is {}"
                      .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, \
                              str(cb_params.net_outputs)), flush=True)
            else:
                print("epoch: {}, current epoch percent: {}, step: {}, loss is {}"
                      .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, \
                              str(cb_params.net_outputs[0])), flush=True)
        else:
            if self.target_device == 'CPU':
                print("epoch: {}, step: {}, loss is {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)
            else:
                print("epoch: {}, step: {}, loss is {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs[0])), flush=True)


class F1:
    '''
    calculate F1 score
    '''
    def __init__(self, num_labels=2, mode="binary"):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.num_labels = num_labels
        self.mode = mode
        if self.mode.lower() not in ("binary", "multilabel"):
            raise ValueError("Assessment mode not supported, support: [Binary, MultiLabel]")
        if self.mode.lower() != "binary":
            self.metric = ConfusionMatrixMetric(skip_channel=False, metric_name=("f1 score"),
                                                calculation_method=False, decrease="mean")

    def update(self, logits, labels):
        """update F1 score"""
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)

        backpointers, best_tag_id = logits
        best_path = postprocess(backpointers, best_tag_id)
        logit_id = []
        for ele in best_path:
            logit_id.extend(ele)

        if self.mode.lower() == "binary":
            pos_eva = np.isin(logit_id, [i for i in range(1, self.num_labels)])
            pos_label = np.isin(labels, [i for i in range(1, self.num_labels)])
            self.TP += np.sum(pos_eva&pos_label)
            self.FP += np.sum(pos_eva&(~pos_label))
            self.FN += np.sum((~pos_eva)&pos_label)
        else:
            target = np.zeros((len(labels), self.num_labels), dtype=np.int)
            pred = np.zeros((len(logit_id), self.num_labels), dtype=np.int)
            for i, label in enumerate(labels):
                target[i][label] = 1
            for i, label in enumerate(logit_id):
                pred[i][label] = 1
            self.metric.update(pred, target)
        return logit_id, labels

    def eval(self):
        return self.metric.eval()


class LSTMCRFLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(LSTMCRFLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(sequence_index, tags_index_map):
    """
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    """
    default = tags_index_map[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags_index_map.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tokens in enumerate(sequence_index):
        # End of a chunk 1
        if tokens == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tokens != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tokens, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(sequence_index))
        chunks.append(chunk)

    return chunks


def get_label_lists(gold_lists, pred_lists, mask_lists=None):
    """get the valid sequence length"""
    preds = list()
    golds = list()
    for gold, pred, mask in zip(gold_lists, pred_lists, mask_lists):
        temp_preds = list()
        temp_golds = list()
        for g, p, m in zip(gold, pred, mask):
            if m == 0:
                continue
            temp_preds.append(p)
            temp_golds.append(g)
        preds.append(temp_preds)
        golds.append(temp_golds)
    return golds, preds
