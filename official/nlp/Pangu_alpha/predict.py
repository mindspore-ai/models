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
PanGu predict run
"""
import json
import os
import requests

import numpy as np
from tqdm import tqdm

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.model import Model
from mindspore.train.serialization import load_distributed_checkpoint
try:
    from mindformers.modules.transformer import TransformerOpParallelConfig
except ImportError as e:
    print("Import `mindformers.modules.transformer` ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)

from src.tokenization_jieba import JIEBATokenizer
from src.generate import get_scores
from src.pangu_alpha import EvalNet, PanguAlphaModel
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import get_args, TimePoint

from tasks import load_metric, load_dataset


def set_auto_parallel_context(args_opt):
    """Set the auto parallel context"""
    rank = 0
    device_num = 1
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    return rank, device_num

def load_model(args_opt):
    r"""
     The main function for load model
    """
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    context.set_context(max_device_memory="30GB")
    # Set parallel context
    rank, device_num = set_auto_parallel_context(args_opt)

    if args_opt.eval_task:
        use_past = False
    else:
        use_past = True if args_opt.export else (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)

    # Set model property, rewrite the model parallel
    if device_num < args_opt.op_level_model_parallel_num:
        print(f"The op_level_model_parallel_num {args_opt.op_level_model_parallel_num} is smaller than the device num，"
              f"so change it to the {device_num}", flush=True)
        args_opt.op_level_model_parallel_num = device_num
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  recompute=False)

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        hidden_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        ffn_hidden_size=args_opt.embedding_size * 4,
        use_past=use_past,
        eod_reset=False,
        parallel_config=parallel_config,
        load_ckpt_path=args_opt.load_ckpt_path,
        run_type=args_opt.run_type,
        param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    # Define network
    pangu_alpha = PanguAlphaModel(config)
    if args_opt.eval_task:
        from src.pangu_alpha import PanGUAlphaLossWithPrompt
        from mindformers import CrossEntropyLoss
        loss = CrossEntropyLoss()
        eval_net = PanGUAlphaLossWithPrompt(config, pangu_alpha, loss)
    else:
        eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif args_opt.eval_task:
        # Compiling only needs the shape
        predict_layout = model_predict.infer_predict_layout(inputs_np, inputs_np, inputs_np)
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_true, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)
    print("======start load_distributed checkpoint", flush=True)
    # For 2.6B and 13B models, the number of ckpt files is 512.
    ckpt_name = 'filerted'
    ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"{ckpt_name}_{ckpt_rank}.ckpt") for ckpt_rank in
                      range(0, 512)]
    print(f"Loading from path {ckpt_file_list[0]}", flush=True)
    # Load checkpoint files
    load_distributed_checkpoint(eval_net, ckpt_file_list, predict_layout)
    print("================load param ok=================", flush=True)
    return model_predict, config


def export_mindir(model_predict, config, export_eval_loss=False):
    """Export mindir model"""

    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    if export_eval_loss:
        print("Start to export the model for task evaluation", flush=True)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        export(model_predict.predict_network, inputs_np, inputs_np,
               file_name='pangu_alpha_1024_eval_loss', file_format='MINDIR')
    else:
        current_index = Tensor(np.array([0]), mstype.int32)

        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        export(model_predict.predict_network, inputs_np, current_index,
               init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        export(model_predict.predict_network, inputs_np_1, current_index,
               init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt):
    """run predict"""
    from src.generate import generate, generate_increment
    # Define tokenizer
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab.model'))

    # Tokenize input sentence to ids
    sample = "今天是一个好天气"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    generate_func = generate_increment if config.use_past else generate
    output_ids = generate_func(model_predict, input_ids, args_opt)
    # Decode output ids to sentence
    output_samples = tokenizer.decode(output_ids.tolist())
    print('Output is:', output_samples, flush=True)


def run_eval(model_predict, config, args_opt):
    """run predict"""
    # Define tokenizer
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab.model'))

    examples = load_dataset(args_opt.eval_task, split='validation', tokenizer=tokenizer,
                            data_url=args_opt.eval_data_url)
    # Tokenize input sentence to ids
    point = TimePoint()
    point.set_start()
    for item in tqdm(examples, total=len(examples)):
        output_ids = get_scores(model_predict, item, tokenizer, pad_length=config.seq_length)
        # Call inference
        item['predict'] = output_ids
    point.set_end()
    # This log cannot be deleted, as the CI monitor this print.
    print(f"Prediction done. Total cost time is {point.get_spend_time()}")
    return examples


def get_query_client(opt):
    """If enable_client is enabled, return the request url for sending the query."""
    server_ip = opt.server_ip
    port = opt.port
    server_name = "pangu"
    format_url = f"http://{server_ip}:{port}/model/{server_name}/version/0:predict"
    print(format_url)
    return format_url


def run_eval_on_server(args_opt):
    """Run evaluation on the given dataset with the server"""
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab.model'))
    examples = load_dataset(args_opt.eval_task, data_url=args_opt.eval_data_url,
                            split='validation', tokenizer=tokenizer)
    url = get_query_client(args_opt)
    # Tokenize input sentence to ids
    point = TimePoint()
    point.set_start()
    for _, item in tqdm(enumerate(examples), total=len(examples)):
        # Call inference
        send_data = json.dumps({"instances": [{"input_sentence": item['input_str'],
                                               "prompt": item['prompt'],
                                               "return_scores": True}]})
        result = requests.post(url, data=send_data)
        result = json.loads(result.text)
        output_samples = result['instances'][0]['output_sentence']
        item['predict'] = output_samples
    point.set_end()
    # This log cannot be deleted, as the CI monitor this print.
    print(f"Prediction done. Total cost time is {point.get_spend_time()}")
    return examples


def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)

    if opt.enable_client:
        examples = run_eval_on_server(opt)
        metric = load_metric(opt.eval_task)(examples)
        print(f"Metric for dataset {opt.eval_task} is {metric}")
        return

    model_predict, config = load_model(opt)

    if opt.export:
        export_mindir(model_predict, config, opt.eval_task != "")
        return

    if opt.eval_task:
        examples = run_eval(model_predict, config, opt)
        metric = load_metric(opt.eval_task)(examples)
        print(f"Metric for dataset {opt.eval_task} is {metric}")
        return

    run_predict(model_predict, config, opt)


if __name__ == "__main__":
    main()
