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
"""
Relation classification train file
"""

import copy

import numpy as np
from mindspore import Model
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import ops as P
from mindspore.communication import get_rank
from mindspore.communication import init
from mindspore.nn import AdamWeightDecay
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor


def load_state_dict(state_dict, config):
    """load state dict"""
    new_state_dict = copy.deepcopy(state_dict)

    for num in range(config.num_hidden_layers):
        for attr_name in ("weight", "bias"):
            if f"encoder.layer.{num}.attention.self1.w2e_query.{attr_name}" not in state_dict:
                new_state_dict[f"encoder.layer.{num}.attention.self1.w2e_query.{attr_name}"] = \
                    state_dict[
                        f"encoder.layer.{num}.attention.self1.query.{attr_name}"
                    ]
            if f"encoder.layer.{num}.attention.self1.e2w_query.{attr_name}" not in state_dict:
                new_state_dict[f"encoder.layer.{num}.attention.self1.e2w_query.{attr_name}"] = \
                    state_dict[
                        f"encoder.layer.{num}.attention.self1.query.{attr_name}"
                    ]
            if f"encoder.layer.{num}.attention.self1.e2e_query.{attr_name}" not in state_dict:
                new_state_dict[f"encoder.layer.{num}.attention.self1.e2e_query.{attr_name}"] = \
                    state_dict[
                        f"encoder.layer.{num}.attention.self1.query.{attr_name}"
                    ]
    return new_state_dict


def get_learning_rate(learning_rate, warmup_steps, max_train_steps):
    lr = []

    for current_step in range(max_train_steps):
        if current_step < warmup_steps:
            warmup_percent = current_step / warmup_steps
            lr.append(warmup_percent)
        else:
            warmup_percent = 1 - current_step / max_train_steps
            lr.append(warmup_percent)

    return Tensor(np.array(lr) * learning_rate, mstype.float32)


# create opt
def _create_optimizer(args, network, dataset_size):
    """create optimizer"""
    all_step = args.num_train_epochs * dataset_size
    lr_schedule = get_learning_rate(args.learning_rate, int(all_step * args.warmup_proportion), all_step)
    param_optimizer = network.trainable_params()
    no_decay = ["bias", "LayerNorm.gamma", "beta"]
    list1 = []
    list2 = []
    for p in param_optimizer:
        flag = True
        for nd in no_decay:
            if nd in p.name:
                list1.append(p)
                flag = False
                break
        if flag:
            list2.append(p)

    optimizer_parameters = [
        {
            "params": list2,
            "weight_decay": args.weight_decay,
        },
        {
            "params": list1,
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamWeightDecay(optimizer_parameters, beta1=args.adam_b1, beta2=args.adam_b2, learning_rate=lr_schedule,
                                eps=args.adam_eps)
    return optimizer


def do_train(dataset=None, network=None, args=None):
    rank = 0
    unsqueeze = P.ExpandDims()
    concat = P.Concat()
    entity_vocab_mask_token = 0
    if args.distribute:
        init()
        rank = get_rank()

    optimizer = _create_optimizer(args, network, dataset.get_dataset_size())
    if args.model_path:
        param_dict = load_checkpoint(args.model_path)
        param_dict = load_state_dict(param_dict, args.model_config)
        args.model_config.vocab_size += 2
        word_emb = param_dict["embeddings.word_embeddings.embedding_table"]
        head_emb = unsqueeze(word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]], 0)
        tail_emb = unsqueeze(word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]], 0)
        param_dict["embeddings.word_embeddings.embedding_table"] = Parameter(
            concat((word_emb, head_emb, tail_emb)))
        entity_emb = param_dict['entity_embeddings.entity_embeddings.embedding_table']
        mask_emb_1 = unsqueeze(entity_emb[entity_vocab_mask_token], 0)
        mask_emb = mask_emb_1.expand_as(Tensor(np.ones((2, mask_emb_1.shape[1]))))
        args.model_config.entity_vocab_size = 3
        param_dict['entity_embeddings.entity_embeddings.embedding_table'] = Parameter(
            concat((entity_emb[:1], mask_emb)))

        load_param_into_net(network, param_dict)

    time_cb = TimeMonitor(200)
    loss_cb = LossMonitor(50)
    config_ck = CheckpointConfig(save_checkpoint_steps=2000, keep_checkpoint_max=15, saved_network=network)

    ckpoint_cb = ModelCheckpoint(prefix='luke_tacred', directory=args.output_dir, config=config_ck)
    network.set_train()

    model = Model(network, optimizer=optimizer)
    cbs = [time_cb, loss_cb]
    if rank == 0:
        cbs = [ckpoint_cb, time_cb, loss_cb]
    model.train(args.num_train_epochs, dataset, callbacks=cbs, dataset_sink_mode=False)
