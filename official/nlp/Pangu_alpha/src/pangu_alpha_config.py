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
network config setting
"""
import mindspore.common.dtype as mstype

class PanguAlphaConfig:
    """
    PanGUConfig config class which defines the model size
    """

    def __init__(self,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=40000,
                 hidden_size=768,
                 ffn_hidden_size=768,
                 num_layers=12,
                 num_heads=12,
                 load_ckpt_path=None,
                 param_init_type=mstype.float32,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 eod_token=6,
                 pad_token=6,
                 use_past=False,
                 hidden_act='fast_gelu',
                 eod_reset=True,
                 enable_offload=False,
                 use_moe=False,
                 expert_num=4,
                 per_token_num_experts_chosen=1,
                 parallel_config=None,
                 run_type='train',
                 softmax_compute_type=mstype.float16):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.eod_token = eod_token
        # Use post-layernorm or pre-layernrom, default:pre-layernorm
        self.post_layernorm_residual = post_layernorm_residual
        self.load_ckpt_path = load_ckpt_path
        self.param_init_type = param_init_type
        self.dropout_rate = dropout_rate
        self.compute_dtype = mstype.float16
        self.parallel_config = parallel_config
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.use_past = use_past
        self.eod_reset = eod_reset
        self.enable_offload = enable_offload
        self.softmax_compute_type = softmax_compute_type
        self.use_moe = bool(use_moe)
        self.expert_num = expert_num
        self.per_token_num_experts_chosen = per_token_num_experts_chosen
        self.run_type = run_type
        self.pad_token = pad_token

    def __str__(self):
        info = '===' * 10 + "[PANGUALPHAConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "--{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info


def set_parse_200B(args_opt):
    r"""
        Set config for 200B mode
    """
    args_opt.embedding_size = 16384
    args_opt.num_layers = 64
    args_opt.num_heads = 128
    if args_opt.per_batch_size == 0:
        args_opt.per_batch_size = 1
    args_opt.word_emb_dp = 0
    if args_opt.run_type == "train":
        args_opt.start_lr = 6e-5
        args_opt.end_lr = 6e-6
        args_opt.stage_num = 16
        args_opt.micro_size = 32
        args_opt.op_level_model_parallel_num = 16
        if args_opt.optimizer_shard == 1:
            args_opt.op_level_model_parallel_num = 8
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 4
        args_opt.micro_size = 1
        args_opt.op_level_model_parallel_num = 16
        if args_opt.optimizer_shard == 1:
            args_opt.op_level_model_parallel_num = 8


def set_parse_13B(args_opt):
    r"""
        Set config for 13B mode
    """
    args_opt.embedding_size = 5120
    args_opt.num_layers = 40
    args_opt.num_heads = 40
    args_opt.word_emb_dp = 1
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 5e-5
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 8
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse_2_6B(args_opt):
    r"""
        Set config for 2.6B mode
    """
    args_opt.embedding_size = 2560
    args_opt.num_layers = 32
    args_opt.num_heads = 32
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 1e-4
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 16
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse_1_3B(args_opt):
    r"""
        Set config for 1.3B mode
    """
    args_opt.embedding_size = 1024
    args_opt.num_layers = 16
    args_opt.num_heads = 32
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 1e-4
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 16
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse(args_opt):
    r"""
        Set config according to the mode
    """
    if args_opt.mode == "self_define":
        return

    parse_fn_dict = {"200B": set_parse_200B, "13B": set_parse_13B, "2.6B": set_parse_2_6B, "1.3B": set_parse_1_3B}
    if args_opt.mode not in parse_fn_dict.keys():
        raise ValueError("Invalid mode: {}. Optional mode: 200B, 13B, 2.6B and 1.3B".format(args_opt.mode))
    parse_fn_dict[args_opt.mode](args_opt)
