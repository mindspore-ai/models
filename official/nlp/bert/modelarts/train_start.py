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

'''
Bert finetune and evaluation script.
'''
import os
import collections
import shutil
import mindspore.common.dtype as mstype
from mindspore import log as logger
from mindspore import Tensor, context, load_checkpoint, export
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_param_into_net

from src.bert_for_finetune import BertSquadCell, BertSquad
from src.dataset import create_squad_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, BertLearningRate, convert_labels_to_index
from src.model_utils.config import config as args_opt, optimizer_cfg, bert_net_cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.finetune_eval_model import BertCLSModel, BertSquadModel, BertNERModel
from src.bert_for_finetune import BertNER

import numpy as np


_cur_dir = os.getcwd()


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                       end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="squad",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = BertSquadCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)


def do_eval(dataset=None, load_checkpoint_path="", eval_batch_size=1):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net = BertSquad(bert_net_cfg, False, 2)
    net.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, segment_ids, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        is_impossible = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                               end_positions, unique_ids, is_impossible)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i in range(eval_batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return output


def modelarts_pre_process():
    '''modelarts pre process function.'''
    args_opt.device_id = get_device_id()
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    args_opt.load_pretrain_checkpoint_path = os.path.join(_file_dir, args_opt.load_pretrain_checkpoint_path)
    args_opt.load_finetune_checkpoint_path = os.path.join(args_opt.output_path, args_opt.load_finetune_checkpoint_path)
    args_opt.save_finetune_checkpoint_path = os.path.join(args_opt.output_path, args_opt.save_finetune_checkpoint_path)
    args_opt.vocab_file_path = os.path.join(args_opt.data_path, args_opt.vocab_file_path)
    if args_opt.schema_file_path:
        args_opt.schema_file_path = os.path.join(args_opt.data_path, args_opt.schema_file_path)
    args_opt.train_data_file_path = os.path.join(args_opt.data_path, args_opt.train_data_file_path)
    args_opt.eval_json_path = os.path.join(args_opt.data_path, args_opt.eval_json_path)

def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def run_export(ckpt_dir):
    '''export function'''
    ckpt_file = _get_last_ckpt(ckpt_dir)
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)

    if args_opt.description == "run_ner":
        label_list = []
        with open(args_opt.label_file_path) as f:
            for label in f:
                label_list.append(label.strip())

        tag_to_index = convert_labels_to_index(label_list)

        if args_opt.use_crf.lower() == "true":
            max_val = max(tag_to_index.values())
            tag_to_index["<START>"] = max_val + 1
            tag_to_index["<STOP>"] = max_val + 2
            number_labels = len(tag_to_index)
            net = BertNER(bert_net_cfg, args_opt.export_batch_size, False, num_labels=number_labels,
                          use_crf=True, tag_to_index=tag_to_index)
        else:
            number_labels = len(tag_to_index)
            net = BertNERModel(bert_net_cfg, False, number_labels, use_crf=(args_opt.use_crf.lower() == "true"))
    elif args_opt.description == "run_classifier":
        net = BertCLSModel(bert_net_cfg, False, num_labels=args_opt.num_class)
    elif args_opt.description == "run_squad":
        net = BertSquadModel(bert_net_cfg, False)
    else:
        raise ValueError("unsupported downstream task")

    load_checkpoint(ckpt_file, net=net)
    net.set_train(False)

    input_ids = Tensor(np.zeros([args_opt.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)
    input_mask = Tensor(np.zeros([args_opt.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)
    token_type_id = Tensor(np.zeros([args_opt.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)
    label_ids = Tensor(np.zeros([args_opt.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)

    if args_opt.description == "run_ner" and args_opt.use_crf.lower() == "true":
        input_data = [input_ids, input_mask, token_type_id, label_ids]
    else:
        input_data = [input_ids, input_mask, token_type_id]
    export(net, *input_data, file_name=args_opt.export_file_name, file_format=args_opt.file_format)
    if args_opt.enable_modelarts:
        air_file = f"{args_opt.export_file_name}.{args_opt.file_format.lower()}"
        shutil.move(air_file, args_opt.output_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_squad():
    """run squad task"""
    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true":
        if args_opt.vocab_file_path == "":
            raise ValueError("'vocab_file_path' must be set when do evaluation task")
        if args_opt.eval_json_path == "":
            raise ValueError("'tokenization_file_path' must be set when do evaluation task")
    epoch_num = args_opt.epoch_num
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(enable_graph_kernel=True)
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    netwithloss = BertSquad(bert_net_cfg, True, 2, dropout_prob=0.1)

    if args_opt.do_train.lower() == "true":
        ds = create_squad_dataset(batch_size=args_opt.train_batch_size,
                                  data_file_path=args_opt.train_data_file_path,
                                  schema_file_path=args_opt.schema_file_path,
                                  do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)
        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir, "squad")

    if args_opt.do_eval.lower() == "true":
        from src import tokenization
        from src.create_squad_data import read_squad_examples, convert_examples_to_features
        from src.squad_get_predictions import write_predictions
        from src.squad_postprocess import SQuad_postprocess
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file_path, do_lower_case=True)
        eval_examples = read_squad_examples(args_opt.eval_json_path, False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=bert_net_cfg.seq_length,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=None,
            vocab_file=args_opt.vocab_file_path)
        ds = create_squad_dataset(batch_size=args_opt.eval_batch_size,
                                  data_file_path=eval_features,
                                  schema_file_path=args_opt.schema_file_path, is_training=False,
                                  do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        outputs = do_eval(ds, load_finetune_checkpoint_path, args_opt.eval_batch_size)
        all_predictions = write_predictions(eval_examples, eval_features, outputs, 20, 30, True)
        SQuad_postprocess(args_opt.eval_json_path, all_predictions, output_metrics="output.json")
    run_export(save_finetune_checkpoint_path)


if __name__ == "__main__":
    run_squad()
    