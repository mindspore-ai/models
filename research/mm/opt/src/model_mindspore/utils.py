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
Utils function for the parallel training.
This is an experimental interface that is subject to change and/or deletion.
"""

from multiprocessing import Process
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
import numpy as np

_get_square_sum = C.MultitypeFuncGraph("_get_square_sum")


@_get_square_sum.register("Tensor", "Number")
def _get_square_sum_func(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


_apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@_apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm_func(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad


def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class _GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(_GlobalNorm, self).__init__()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = (config.pipeline_stage > 1)
        if self.is_pipeline:
            group_size = config.mp
            group_list, group_name = _get_model_parallel_group(config.mp)
            create_group(group_name, group_list)
            self.allreduce = P.AllReduce(group=group_name)
            pipeline_group_list, pipeline_group_name = _get_pipeline_group()
            create_group(pipeline_group_name, pipeline_group_list)
            self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        else:
            group_size = get_group_size()
        if config.vocab_emb_dp:
            self.allreduce_filter = tuple("projection.bias" not in x.name and "layernorm" not in x.name
                                          and "embedding_table" not in x.name for x in params)
        else:
            self.allreduce_filter = tuple("projection.bias" not in x.name and "layernorm" not in x.name
                                          and "position_embedding.embedding_table" not in x.name for x in params)
        self.allreduce_group_size = ()

        self.init_params(params, config, group_size)


    def init_params(self, params, config, group_size):
        """ init_params """

        for x in params:
            if "uniter.encoder" in x.name:
                if "dense" in x.name and "weight" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (1.0,)
                elif "projection" in x.name and "weight" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (1.0,)
                elif "wi" in x.name or "wo" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (1.0,)
                elif "dense" in x.name and "bias" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (config.dp * 1.0,)
                else:
                    self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)
            elif "txt_output" in x.name or "img_output" in x.name:
                if "weight" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (config.dp * 1.0,)
                elif "dense" in x.name and "bias" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (config.dp * 1.0,)
                elif "mapping" in x.name and "bias" in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (config.dp * 1.0,)
                else:
                    self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)
            else:
                self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(_get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        if self.is_pipeline:
            stage_square_reduce_sum = self.allreduce(square_reduce_sum)
            global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
            global_norms = F.sqrt(global_square_reduce_sum)
        else:
            global_norms = F.sqrt(P.AllReduce()(square_reduce_sum))
        return global_norms


class _ClipByGlobalNorm(nn.Cell):
    """
        Clip grads by global norm
    """

    def __init__(self, params, parallel_config, clip_norm=1.0):
        super(_ClipByGlobalNorm, self).__init__()
        # According to the parallel mode, enabling the parallel global norm or not
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.global_norm = _GlobalNorm(params, parallel_config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        """Clip grads by global norm construct"""
        global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(_apply_global_norm, self.clip_norm, global_norm), grads)
        return grads


class LearningRate(LearningRateSchedule):
    """
        Learning_rate sheduler
    """

    def __init__(self,
                 start_learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(start_learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(start_learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, start_learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """Learning_rate sheduler construct"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class LossSummaryCallback(Callback):
    """
    LossSummaryCallback
    """
    def __init__(self, summary_dir, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                 bucket='obs://mindspore-file/loss_file/summary/', syn_times=100):

        import moxing as mox

        self._summary_dir = summary_dir
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.bucket = bucket
        self.syn_times = syn_times
        self.id2task = {
            0: 'mlmThree',
            1: 'mrcThree',
            2: 'mrfrThree',
            3: 'mafrThree',
            4: 'macThree',
            5: "itmThree",
            6: 'mrctThree',
            7: "tdThree",
            8: "idThree"
        }

        if not mox.file.exists(self.bucket):
            print("++++++++++Creating summary bueckt dir {}".format(self.bucket))
            mox.file.make_dirs(self.bucket)

        print("++++++++++entering")
        self.summary_record = SummaryRecord(self._summary_dir)

    def step_end(self, run_context):
        """
        step_end
        """

        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        # create a confusion matric image, and record it to summary file
        print("++++++++++writing")
        task_id = cb_params.net_outputs[3].asnumpy()[0]
        self.summary_record.add_value('scalar', 'scale', cb_params.net_outputs[2])
        self.summary_record.add_value('scalar', '{}'.format(self.id2task[task_id]), cb_params.net_outputs[0])
        self.summary_record.record(cur_step)

        print("writing finished...", cur_step, self.syn_times)
        if cur_step % self.syn_times == 0:
            print("++++++++++Copying summary to the bueckets start", flush=True)
            self.summary_record.flush()
            self.syn_files()
            print("++++++++++Copying summary to the bueckets ends", flush=True)

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self._summary_dir, self.bucket), name="file_sync")
        process.start()






class LossSummaryCallbackLocal(Callback):
    """
    LossSummaryCallbackLocal
    """
    def __init__(self, summary_dir, local_rank=0, has_trained_epoch=0, has_trained_step=0):


        self._summary_dir = summary_dir
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.id2task = {
            0: 'mlmThree',
            1: 'mrcThree',
            2: 'mrfrThree',
            3: 'mafrThree',
            4: 'macThree',
            5: "itmThree",
            6: 'mrctThree',
            7: "tdThree",
            8: "idThree"
        }

        print("++++++++++entering")
        self.summary_record = SummaryRecord(self._summary_dir)

    def step_end(self, run_context):
        """
        step_end
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        # create a confusion matric image, and record it to summary file
        print("++++++++++writing")
        self.summary_record.add_value('scalar', 'loss', cb_params.net_outputs[0])
        self.summary_record.record(cur_step)
