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
network config setting, gradient clip function and dynamic learning rate function
"""
from multiprocessing import Process
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord

import moxing as mox



class GPTConfig:
    """
    GPT config class which defines the model size
    """

    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 self_layernorm=True,
                 forward_reduce_scatter=True,
                 word_emb_dp=True,
                 stage_num=16,
                 micro_size=32,
                 eod_reset=True,
                 use_top_query_attention=True):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        self.use_past = use_past
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        self.self_layernorm = self_layernorm
        self.forward_reduce_scatter = forward_reduce_scatter
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp
        self.eod_reset = eod_reset
        self.use_top_query_attention = use_top_query_attention

    def __str__(self):
        info = "[GPTConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.allreduce_filter = tuple(
            "projection.bias" not in x.name and
            "layernorm" not in x.name and "position_embedding.embedding_table" not in x.name
            for x in params)
        self.allreduce_group_size = ()
        for item in self.allreduce_filter:
            if item:
                self.allreduce_group_size = self.allreduce_group_size + (1.0,)
            else:
                self.allreduce_group_size = self.allreduce_group_size + (config.mp * 1.0,)
        self.length = len(params)
        group_list, group_name = _get_model_parallel_group(config.mp)
        print("rank_list", group_name)
        print("group_size_list", self.allreduce_group_size)
        create_group(group_name, group_list)
        self.allreduce = P.AllReduce(group=group_name)
        pipeline_group_list, pipeline_group_name = _get_pipeline_group()
        print("pipeline_group_name", pipeline_group_name)
        create_group(pipeline_group_name, pipeline_group_list)
        self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        self.group_name1 = group_name
        self.group_name2 = pipeline_group_name

    def construct(self, grads):
        square_sum = self.hyper_map(
            get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        stage_square_reduce_sum = self.allreduce(square_reduce_sum)
        global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
        global_norms = F.sqrt(global_square_reduce_sum)
        return global_norms


class GlobalNormOptShard(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(GlobalNormOptShard, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        device_nums = get_group_size()
        per_stage_device_num = device_nums // config.stage_num
        self.allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "embedding_table" not in x.name:
                self.allreduce_group_size = self.allreduce_group_size + (1.0,)
            elif "position_embedding.embedding_table" not in x.name and "projection.bias" not in x.name:
                self.allreduce_group_size = self.allreduce_group_size + (config.dp * 1.0,)
            else:
                self.allreduce_group_size = self.allreduce_group_size + (per_stage_device_num * 1.0,)
        self.length = len(params)
        group_list, group_name = _get_model_parallel_group(per_stage_device_num)
        print("rank_list", group_name)
        print("group_size_list", self.allreduce_group_size)
        create_group(group_name, group_list)
        self.allreduce = P.AllReduce(group=group_name)
        pipeline_group_list, pipeline_group_name = _get_pipeline_group()
        print("pipeline_group_name", pipeline_group_name)
        create_group(pipeline_group_name, pipeline_group_list)
        self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        self.group_name1 = group_name
        self.group_name2 = pipeline_group_name

    def construct(self, grads):
        square_sum = self.hyper_map(
            get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        stage_square_reduce_sum = self.allreduce(square_reduce_sum)
        global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
        global_norms = F.sqrt(global_square_reduce_sum)
        return global_norms


class ClipByGlobalNorm(nn.Cell):
    """
    Clip grads by global norm
    """

    def __init__(self, params, config, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm_origin = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_origin, self.clip_norm)
        global_norm = F.select(cond, global_norm_origin, self.clip_norm)
        grads = self.hyper_map(
            F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_origin


class ClipByGlobalNormOptShard(nn.Cell):
    """
    Clip grads by global norm
    """

    def __init__(self, params, config, clip_norm=1.0):
        super(ClipByGlobalNormOptShard, self).__init__()
        self.global_norm = GlobalNormOptShard(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm_origin = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_origin, self.clip_norm)
        global_norm = F.select(cond, global_norm_origin, self.clip_norm)
        grads = self.hyper_map(
            F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_origin


def _get_model_parallel_group(mp):
    """get model parallel group"""
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    if len(rank_str_list) < 30:
        rank_list_str = "-".join(rank_str_list)
    else:
        rank_list_str = rank_str_list[0] + "-to-" + rank_str_list[len(rank_str_list) - 1] + "-" + str(
            len(rank_str_list))
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """get pipeline group"""
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x *
                 per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x *
                         per_stage_device_nums) for x in group]
    if len(rank_str_list) < 30:
        rank_list_str = "-".join(rank_str_list)
    else:
        rank_list_str = rank_str_list[0] + "-to-" + rank_str_list[len(rank_str_list) - 1] + "-" + str(
            len(rank_str_list))
    return rank_list, rank_list_str


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for GPT network.
    """

    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True,
                 lr_scale=0.125):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine
        self.lr_scale = lr_scale

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr * self.lr_scale


class LossSummaryCallback(Callback):
    """Loss Summary Callback"""
    def __init__(self, summary_dir, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                 bucket='obs://mindspore-file/loss_file/summary/', syn_times=100):
        self._summary_dir = summary_dir
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.bucket = bucket
        self.syn_times = syn_times

        if not mox.file.exists(self.bucket):
            print("Creating summary bueckt dir {}".format(self.bucket))
            mox.file.make_dirs(self.bucket)

        print("entering")
        self.summary_record = SummaryRecord(self._summary_dir)

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        # create a confusion matric image, and record it to summary file
        print("writing")
        self.summary_record.add_value('scalar', 'loss', cb_params.net_outputs[0])
        self.summary_record.add_value('scalar', 'scale', cb_params.net_outputs[2])
        if len(cb_params.net_outputs) > 3:
            self.summary_record.add_value('scalar', 'global_norm', cb_params.net_outputs[3])
        self.summary_record.record(cur_step)

        print("writing finished...", cur_step, self.syn_times)
        if cur_step % self.syn_times == 0:
            print("Copying summary to the bueckets start", flush=True)
            self.summary_record.flush()
            self.syn_files()
            print("Copying summary to the bueckets ends", flush=True)

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(
            self._summary_dir, self.bucket), name="file_sync")
        process.start()
        # mox.file.copy_parallel(self._summary_dir, self.bucket)



class StrategyCkptCallback(Callback):
    """Strategy Ckpt Callback"""
    def __init__(self, strategy_file, local_rank=0, bucket='s3://muti-modal/strategy_ckpt/opt/'):
        self._strategy_file = strategy_file
        self.local_rank = local_rank
        self.bucket = bucket + str(local_rank) + "/"
        self.obs_file = self.bucket + "strategy" + str(local_rank) + ".ckpt"
        self.has_synced = False
        if not mox.file.exists(self.bucket):
            print("Creating strategy ckpt bueckt dir {}".format(self.bucket))
            mox.file.make_dirs(self.bucket)

    def step_end(self, run_context):
        if not self.has_synced:
            print("Copying strategy_ckpt to the bueckets start", flush=True)
            self.syn_files()
            print("Copying strategy_ckpt to the bueckets ends", flush=True)
            self.has_synced = True

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(
            self._strategy_file, self.obs_file), name="file_sync")
        process.start()
