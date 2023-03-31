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
PanguAlpha train script
"""

import datetime
import json
import glob
import os
import math

from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net

try:
    from mindformers.core import CrossEntropyLoss
    from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)

from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha import PanGUAlphaWithLoss, PanguAlphaModel
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from src.callbacks import EvalCallBack, LossCallBack
from src.metrics import PPLMetric

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def add_checkpoint_callback_policy(args_param, callback, rank_id):
    r"""
    Add checkpoint policy to callback.
    """
    if args_param.save_checkpoint:
        # checkpoint store epoch_num and step_num info
        ckpt_append_info = [{"epoch_num": args_param.has_trained_epoches, "step_num": args_param.has_trained_steps}]
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args_param.save_checkpoint_steps,
                                       keep_checkpoint_max=args_param.keep_checkpoint_max,
                                       integrated_save=False,
                                       append_info=ckpt_append_info
                                       )

        # save checkpoint into rank directory
        ckpoint_cb = ModelCheckpoint(prefix=args_param.ckpt_name_prefix + str(rank_id),
                                     directory=os.path.join(args_param.save_checkpoint_path, f"rank_{rank_id}"),
                                     config=ckpt_config)

        callback.append(ckpoint_cb)


def set_parallel_context(args_opt):
    r"""Set parallel context"""
    D.init()
    device_num = D.get_group_size()
    rank = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank, device_num))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        parallel_mode=args_opt.parallel_mode, gradients_mean=False, search_mode=args_opt.search_mode,
        full_batch=bool(args_opt.full_batch), strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
        enable_parallel_optimizer=bool(args_opt.optimizer_shard), strategy_ckpt_save_file='strategy.ckpt',
        enable_alltoall=bool(args_opt.enable_alltoall))
    set_algo_parameters(elementwise_op_strategy_follow=True)
    if context.get_auto_parallel_context("parallel_mode") == ParallelMode.AUTO_PARALLEL:
        set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    _set_multi_subgraphs()
    return rank, device_num


def set_optimizer(optimizer, opt_offload, group_params, learning_rate, config):
    r"""Set optimizer"""
    if optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=learning_rate)
    elif opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=learning_rate, eps=1e-8, beta1=0.9, beta2=0.95,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=learning_rate, eps=1e-8, beta1=0.9, beta2=0.95)
    return optimizer


def cal_model_property(args_opt, device_num):
    # in case when the model parallel is smaller than the device num, the model_parallel_num will be zero.
    model_parallel_num = min(args_opt.op_level_model_parallel_num, device_num)
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    if (context.get_auto_parallel_context("parallel_mode") == ParallelMode.DATA_PARALLEL or
            context.get_auto_parallel_context("parallel_mode") == ParallelMode.AUTO_PARALLEL):
        batch_size = args_opt.per_batch_size
    return model_parallel_num, data_parallel_num, batch_size


def run_train(args_opt):
    r"""The main training process."""
    # Set execution mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, max_device_memory="30GB")
    # Set parallel context
    rank = 0
    device_num = 1
    if args_opt.distribute == "true":
        rank, device_num = set_parallel_context(args_opt)
    context.set_context(save_graphs=False, save_graphs_path="./graphs_of_device_id_" + str(rank))
    if args_opt.parallel_mode == "data_parallel":
        # in avoid of the loop call depth
        context.set_context(max_call_depth=10000)

    # env variable prepare
    group_info_file = os.getenv("GROUP_INFO_FILE")
    if group_info_file:
        with open(os.path.expanduser("job/code/group_info_env"), "a") as outfile:
            outfile.write(f"export GROUP_INFO_FILE_REFLECT={group_info_file}\n")

    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    eval_cache_url = '/cache/EvalData/'
    if args_opt.offline:
        cache_url = args_opt.data_url
        eval_cache_url = args_opt.eval_data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank)
        download_data(src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank)
    # Set model property
    model_parallel_num, data_parallel_num, batch_size = cal_model_property(args_opt, device_num)
    micro_batch_interleaved = args_opt.micro_batch_interleaved
    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(args_opt.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  expert_parallel=args_opt.expert_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=bool(args_opt.optimizer_shard),
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp), recompute=recompute_config,
                                                  gradient_aggregation_group=args_opt.gradient_aggregation_group)
    config = PanguAlphaConfig(batch_size=batch_size // micro_batch_interleaved, num_heads=args_opt.num_heads,
                              hidden_size=args_opt.embedding_size, seq_length=args_opt.seq_length,
                              vocab_size=args_opt.vocab_size, num_layers=args_opt.num_layers,
                              ffn_hidden_size=args_opt.embedding_size * 4, eod_reset=bool(args_opt.eod_reset),
                              load_ckpt_path=args_opt.load_ckpt_path, expert_num=args_opt.expert_num,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              enable_offload=bool(args_opt.opt_offload), use_moe=bool(args_opt.use_moe),
                              per_token_num_experts_chosen=args_opt.per_token_num_experts_chosen,
                              hidden_act='fast_gelu' if args_opt.device_target != "GPU" else 'gelu',
                              parallel_config=parallel_config, eod_token=args_opt.eod_id)
    print("===config is: ", config, flush=True)
    # Define network
    pangu_alpha = PanguAlphaModel(config=config)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    pangu_alpha_with_loss_net = MicroBatchInterleaved(PanGUAlphaWithLoss(config, pangu_alpha, loss),
                                                      micro_batch_interleaved)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss_net)
    print("=====args_opt is: ", args_opt, flush=True)
    # Warm-up and cosine decay learning rate
    lr = LearningRate(learning_rate=args_opt.start_lr, end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step, decay_steps=200000)
    params = pangu_alpha_with_loss.trainable_params()
    group_params = set_weight_decay(params)
    optimizer = set_optimizer(args_opt.optimizer, args_opt.opt_offload, group_params=group_params,
                              learning_rate=lr, config=config)
    epoch_num = args_opt.epoch_size
    # Dataset loading mindrecord files
    ds = create_dataset(config.batch_size * micro_batch_interleaved, data_path=cache_url, data_start_index=0,
                        eod_reset=config.eod_reset, full_batch=bool(args_opt.full_batch), eod_id=args_opt.eod_id,
                        device_num=device_num, rank=rank, column_name=args_opt.data_column_name, epoch=epoch_num)
    step_per_epoch = ds.get_dataset_size()
    actual_epoch_num = int(epoch_num * step_per_epoch / args_opt.sink_size)
    loss_callback = LossCallBack(step_per_epoch, rank, 0, 0, micro_size=micro_batch_interleaved)
    callback = [TimeMonitor(args_opt.sink_size), loss_callback]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=math.pow(2, 32), scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, scale_update_cell=update_cell, enable_global_norm=True,
        config=config)
    if args_opt.train_and_eval_mode:
        ds_eval = create_dataset(config.batch_size * micro_batch_interleaved, data_path=eval_cache_url,
                                 data_start_index=0, eod_reset=config.eod_reset, full_batch=bool(args_opt.full_batch),
                                 eod_id=args_opt.eod_id, device_num=device_num, rank=rank,
                                 column_name=args_opt.data_column_name, epoch=epoch_num,
                                 num_samples=args_opt.eval_steps * config.batch_size)
        ppl_metric = PPLMetric(config.seq_length)
        model = Model(pangu_alpha_with_grads, eval_network=pangu_alpha_with_loss, metrics={"ppl": ppl_metric})
        callback.append(EvalCallBack(model, ds_eval, ppl_metric))
    else:
        model = Model(pangu_alpha_with_grads)
    if args_opt.pre_trained:
        flag = restore_exception_checkpoint(args_opt, args_opt.sink_size, ds, model,
                                            pangu_alpha_with_grads, epoch=actual_epoch_num)
        if not flag:
            restore_checkpoint(args_opt, args_opt.sink_size, ds, model,
                               pangu_alpha_with_grads, epoch=actual_epoch_num)
        loss_callback.has_trained_epoch = args_opt.has_trained_epoches
        loss_callback.has_trained_step = args_opt.has_trained_steps
    add_checkpoint_callback_policy(args_opt, callback, rank)
    if args_opt.incremental_training:
        strategy = model.infer_train_layout(train_dataset=ds, sink_size=args_opt.sink_size)
        print("======start load_distributed checkpoint", flush=True)
        # For 2.6B and 13B models, the number of ckpt files is 512.
        ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"filerted_{ckpt_rank}.ckpt") for ckpt_rank in
                          range(0, 512)]
        print(f"Loading from path {ckpt_file_list[0]}", flush=True)
        load_distributed_checkpoint(model.train_network, ckpt_file_list, strategy)
    print("Dataset size: {}, actual_epoch_num: {}".format(step_per_epoch, actual_epoch_num), flush=True)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=args_opt.sink_size, dataset_sink_mode=True)


def restore_checkpoint(args_param, sink_size, dataset, model, network, epoch):
    r"""
    Load checkpoint process.
    """
    print("======start single checkpoint", flush=True)
    ckpt_name = args_param.ckpt_name_prefix
    ckpt_pattern = os.path.join(args_param.save_checkpoint_path, "rank_{}".format(D.get_rank()),
                                f"{ckpt_name}*.ckpt")
    ckpt_all_files = glob.glob(ckpt_pattern)

    if not ckpt_all_files:
        print(f"There is no ckpt file in {args_param.save_checkpoint_path}, "
              f"current ckpt_files found is {ckpt_all_files} "
              f"with pattern {ckpt_pattern}, so skip the loading.")

    ckpt_exp_pattern = os.path.join(args_param.save_checkpoint_path, "rank_{}".format(D.get_rank()),
                                    f"{ckpt_name}*_breakpoint.ckpt")
    ckpt_exp_files = glob.glob(ckpt_exp_pattern)
    ckpt_files = []
    for file in ckpt_all_files:
        if file not in ckpt_exp_files:
            ckpt_files.append(file)

    if not ckpt_files:
        print(f"There is no ckpt file in {args_param.save_checkpoint_path}, "
              f"current ckpt_files found is {ckpt_files} "
              f"with pattern {ckpt_pattern}, so skip the loading.")
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    time_stamp = datetime.datetime.now()
    print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')} pre trained ckpt model {ckpt_files} loading",
          flush=True)
    # Load checkpoint files latest file
    print(f'Start to load from {ckpt_files[0]}')
    param_dict = load_checkpoint(ckpt_files[0])
    if param_dict.get("epoch_num") and param_dict.get("step_num"):
        args_param.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        args_param.has_trained_step = int(param_dict["step_num"].data.asnumpy())
    model.build(train_dataset=dataset, sink_size=sink_size, epoch=epoch)
    load_param_into_net(network, param_dict)


def get_exception_checkpoints(args_param):
    r"""
    Load checkpoint process.
    """
    print("======start exception checkpoint", flush=True)
    restore_ranks = os.getenv("RESTORE_RANKS")
    if not restore_ranks:
        return None

    restore_rank_list = list(map(int, restore_ranks.split(",")))
    ckpt_file_list = []
    ckpt_name = args_param.ckpt_name_prefix
    for ckpt_rank in restore_rank_list:
        ckpt_pattern = os.path.join(args_param.save_checkpoint_path,
                                    f"rank_{ckpt_rank}",
                                    f"{ckpt_name}*_breakpoint.ckpt")
        ckpt_files = glob.glob(ckpt_pattern)
        if not ckpt_files:
            print(
                f"There is no ckpt file in {args_param.save_checkpoint_path}, "
                f"current ckpt_files found is {ckpt_files} "
                f"with pattern {ckpt_pattern}, so skip the loading.")
            return None
        ckpt_files.sort(key=os.path.getmtime, reverse=True)
        ckpt_file_list.append(ckpt_files[0])
    print(f"checkpoint file {ckpt_file_list}")
    return ckpt_file_list


def check_exception_checkpoints(ckpt_file_list):
    """
    Check exception checkpoints size.
    Args:
        ckpt_file_list: exception checkpoints

    Returns: result of exception checkpoints size check.

    """
    ckpt_size_list = []
    for ckpt_file in ckpt_file_list:
        ckpt_size_list.append(os.path.getsize(ckpt_file))

    if len(set(ckpt_size_list)) > 1:
        return False

    return True


def restore_exception_checkpoint(args_param, sink_size, dataset, model, network, epoch):
    """
    Restore exception checkpoint to training model.
    Args:
        args_param: model training parameters
        sink_size: model training sink size
        dataset: dataset used for training
        model: model
        network: pangu_alpha network
        epoch: training epoch

    Returns: load exception checkpont success or not.

    """
    if os.getenv("RESTORE_RANKS") == "-1":
        return False

    ckpt_file_list = get_exception_checkpoints(args_param)

    restore_flag = False
    if ckpt_file_list:
        restore_flag = check_exception_checkpoints(ckpt_file_list)

    if not restore_flag:
        return False

    ckpt_name = args_param.ckpt_name_prefix
    restore_ranks_map = os.getenv("RESTORE_RANKS_MAP")
    if not restore_ranks_map:
        return False

    try:
        print("whether run into load process")
        restore_ranks_map_json = json.loads(restore_ranks_map)
        map_rank_id = D.get_rank()
        for key in restore_ranks_map_json.keys():
            key_list = list(key.split(","))
            if str(D.get_rank()) in key_list:
                map_rank_id = restore_ranks_map_json.get(key)

        print(f"loading map rank id {map_rank_id}")
        ckpt_pattern = os.path.join(args_param.save_checkpoint_path,
                                    f"rank_{map_rank_id}",
                                    f"{ckpt_name}*breakpoint.ckpt")
        ckpt_files = glob.glob(ckpt_pattern)
        ckpt_files.sort(key=os.path.getmtime, reverse=True)
        print(f" checkpoint files {ckpt_files[0]}")
        param_dict = load_checkpoint(ckpt_files[0])
        print(f" checkpoint param dict epoch num {param_dict.get('epoch_num')}")
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            args_param.has_trained_epoches = int(
                param_dict["epoch_num"].data.asnumpy())
            args_param.has_trained_steps = int(
                param_dict["step_num"].data.asnumpy())

        # Load checkpoint files
        model.build(train_dataset=dataset, sink_size=sink_size, epoch=epoch)
        load_param_into_net(network, param_dict)
    except TypeError:
        return False
    else:
        return True


def set_pipeline_parallel_context(args_opt):
    r"""Set pipeline parallel context."""
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank_id, device_num))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        parallel_mode=args_opt.parallel_mode, gradients_mean=False, search_mode=args_opt.search_mode,
        full_batch=bool(args_opt.full_batch), loss_repeated_mean=True,
        device_num=device_num, enable_parallel_optimizer=bool(args_opt.optimizer_shard),
        pipeline_stages=args_opt.stage_num, enable_alltoall=bool(args_opt.enable_alltoall))
    set_algo_parameters(elementwise_op_strategy_follow=True)
    if context.get_auto_parallel_context("parallel_mode") == ParallelMode.AUTO_PARALLEL:
        set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    _set_multi_subgraphs()
    return rank_id, device_num


def cal_model_property_pipeline(args_opt, device_num):
    is_auto_parallel = context.get_auto_parallel_context("parallel_mode") == ParallelMode.AUTO_PARALLEL
    # in order to make sure data_parallel_num is always non-zero, set model_parallel_num to 1
    model_parallel_num = 1 if is_auto_parallel else args_opt.op_level_model_parallel_num
    stage_device_num = int(device_num / args_opt.stage_num)
    data_parallel_num = int(stage_device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num * args_opt.micro_size
    if is_auto_parallel:
        batch_size = per_batch_size * args_opt.micro_size
    return model_parallel_num, data_parallel_num, batch_size


def run_train_pipeline(args_opt):
    r"""The main training process in pipeline."""
    context.set_context(save_graphs=False, mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(max_device_memory="30GB")
    rank_id = 0
    device_num = 1
    if args_opt.distribute == "true":
        rank_id, device_num = set_pipeline_parallel_context(args_opt)
    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    eval_cache_url = '/cache/EvalData/'
    if args_opt.offline:
        cache_url = args_opt.data_url
        eval_cache_url = args_opt.eval_data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank_id)
        download_data(src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank_id)
    model_parallel_num, data_parallel_num, batch_size = cal_model_property_pipeline(args_opt, device_num)
    stage_device_num = int(device_num / args_opt.stage_num)
    is_last_stage = (rank_id // stage_device_num) == args_opt.stage_num -1
    micro_batch_interleaved = args_opt.micro_batch_interleaved
    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(args_opt.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=bool(args_opt.optimizer_shard),
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp), recompute=recompute_config)
    config = PanguAlphaConfig(batch_size=batch_size // parallel_config.micro_batch_num // micro_batch_interleaved,
                              num_heads=args_opt.num_heads, hidden_size=args_opt.embedding_size,
                              seq_length=args_opt.seq_length, vocab_size=args_opt.vocab_size,
                              use_moe=bool(args_opt.use_moe), eod_token=args_opt.eod_id,
                              num_layers=args_opt.num_layers, ffn_hidden_size=args_opt.embedding_size * 4,
                              eod_reset=bool(args_opt.eod_reset), load_ckpt_path=args_opt.load_ckpt_path,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              enable_offload=bool(args_opt.opt_offload), parallel_config=parallel_config)

    print("[Configure] is: ", config, flush=True)
    pangu_alpha = PanguAlphaModel(config=config)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    pangu_alpha_with_loss_net = PipelineCell(MicroBatchInterleaved(PanGUAlphaWithLoss(config, pangu_alpha, loss),
                                                                   micro_batch_interleaved),
                                             config.parallel_config.micro_batch_num)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss_net)
    print("[args_opt] is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr, end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step, decay_steps=args_opt.decay_steps)
    params = pangu_alpha_with_loss.trainable_params()
    group_params = set_weight_decay(params)
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif args_opt.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    ds = create_dataset(config.batch_size * parallel_config.micro_batch_num * micro_batch_interleaved,
                        data_path=cache_url, device_num=stage_device_num,
                        rank=rank_id % stage_device_num, eod_reset=True, data_start_index=0,
                        full_batch=context.get_auto_parallel_context("full_batch"),
                        column_name=args_opt.data_column_name)
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    loss_callback = LossCallBack(step_per_epoch, rank_id, is_last_stage=is_last_stage,
                                 micro_size=parallel_config.micro_batch_num * micro_batch_interleaved)
    callback = [TimeMonitor(callback_size), loss_callback]
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainPipelineWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)
    if args_opt.train_and_eval_mode:
        ds_eval = create_dataset(config.batch_size * parallel_config.micro_batch_num * micro_batch_interleaved,
                                 data_path=eval_cache_url,
                                 device_num=stage_device_num, rank=rank_id % stage_device_num, eod_reset=True,
                                 data_start_index=0, full_batch=bool(args_opt.full_batch),
                                 column_name=args_opt.data_column_name,
                                 num_samples=args_opt.eval_steps * config.batch_size)
        ppl_metric = PPLMetric(config.seq_length)
        pangu_alpha_with_loss_eval_net = _VirtualDatasetCell(PanGUAlphaWithLoss(config, pangu_alpha, loss))
        model = Model(pangu_alpha_with_grads, eval_network=pangu_alpha_with_loss_eval_net, metrics={"ppl": ppl_metric})
        model.build(ds, ds_eval, sink_size=callback_size)
        eval_callback = EvalCallBack(model, ds_eval, ppl_metric)
        callback.append(eval_callback)
    else:
        model = Model(pangu_alpha_with_grads)

    if args_opt.pre_trained:
        flag = restore_exception_checkpoint(args_opt, callback_size, ds, model,
                                            pangu_alpha_with_grads, epoch=actual_epoch_num)
        if not flag:
            restore_checkpoint(args_opt, callback_size, ds, model, pangu_alpha_with_grads, epoch=actual_epoch_num)

        loss_callback.has_trained_epoch = args_opt.has_trained_epoches
        loss_callback.has_trained_step = args_opt.has_trained_steps
    add_checkpoint_callback_policy(args_opt, callback, rank_id)

    model.train(actual_epoch_num, ds, callbacks=callback,
                sink_size=callback_size, dataset_sink_mode=True)


if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    if opt.per_batch_size == 0:
        raise ValueError("The per_batch_size has not been configured.")
    if bool(opt.enable_alltoall) is True and bool(opt.use_moe) is False:
        raise ValueError("The alltoall communication is only effective when applying moe")
    os.environ['HCCL_CONNECT_TIMEOUT'] = str(opt.hccl_connect_time)
    if opt.stage_num > 1:
        run_train_pipeline(opt)
    else:
        run_train(opt)
