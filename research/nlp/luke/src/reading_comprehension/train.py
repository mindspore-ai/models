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
"""reading comprehension train file"""
import copy
import os

from mindspore.communication import get_rank
from mindspore.nn import AdamWeightDecay, TrainOneStepCell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
import mindspore.ops as P
from mindspore import dtype as mstype
from mindspore._checkparam import Validator as validator

from src.reading_comprehension.model import LukeForReadingComprehensionWithLoss


def load_state_dict(state_dict, config):
    """load state dict"""
    new_state_dict = copy.deepcopy(state_dict)
    if 'adam_v.luke.encoder.layer.0.attention.self1.query.weight' in new_state_dict:
        for num in range(config.num_hidden_layers):
            for attr_name in ("weight", "bias"):
                if f"adam_v.luke.encoder.layer.{num}.attention.self1.w2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"adam_v.luke.encoder.layer.{num}.attention.self1.w2e_query.{attr_name}"] = \
                        state_dict[
                            f"adam_v.luke.encoder.layer.{num}.attention.self1.query.{attr_name}"
                        ]
                if f"adam_v.luke.encoder.layer.{num}.attention.self1.e2w_query.{attr_name}" not in state_dict:
                    new_state_dict[f"adam_v.luke.encoder.layer.{num}.attention.self1.e2w_query.{attr_name}"] = \
                        state_dict[
                            f"adam_v.luke.encoder.layer.{num}.attention.self1.query.{attr_name}"
                        ]
                if f"adam_v.luke.encoder.layer.{num}.attention.self1.e2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"adam_v.luke.encoder.layer.{num}.attention.self1.e2e_query.{attr_name}"] = \
                        state_dict[
                            f"adam_v.luke.encoder.layer.{num}.attention.self1.query.{attr_name}"
                        ]
    else:
        for num in range(config.num_hidden_layers):
            for attr_name in ("weight", "bias"):
                if f"luke.encoder.layer.{num}.attention.self1.w2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"luke.encoder.layer.{num}.attention.self1.w2e_query.{attr_name}"] = \
                        state_dict[
                            f"luke.encoder.layer.{num}.attention.self1.query.{attr_name}"
                        ]
                if f"luke.encoder.layer.{num}.attention.self1.e2w_query.{attr_name}" not in state_dict:
                    new_state_dict[f"luke.encoder.layer.{num}.attention.self1.e2w_query.{attr_name}"] = \
                        state_dict[
                            f"luke.encoder.layer.{num}.attention.self1.query.{attr_name}"
                        ]
                if f"luke.encoder.layer.{num}.attention.self1.e2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"luke.encoder.layer.{num}.attention.self1.e2e_query.{attr_name}"] = \
                        state_dict[
                            f"luke.encoder.layer.{num}.attention.self1.query.{attr_name}"
                        ]
    return new_state_dict


class CustomWarmUpLR(LearningRateSchedule):
    """
    apply the functions to  the corresponding input fields.
    ·
    """

    def __init__(self, learning_rate, warmup_steps, max_train_steps):
        super(CustomWarmUpLR, self).__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
        validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.cast = P.Cast()

    def construct(self, current_step):
        if current_step < self.warmup_steps:
            warmup_percent = self.cast(current_step, mstype.float32) / self.warmup_steps
        else:
            warmup_percent = 1 - self.cast(current_step, mstype.float32) / self.max_train_steps

        return self.learning_rate * warmup_percent


# create opt
def _create_optimizer(args, network, dataset_size):
    """create optimizer"""
    all_step = args.num_train_epochs * dataset_size
    lr_schedule = CustomWarmUpLR(learning_rate=args.learning_rate,
                                 warmup_steps=int(all_step * args.warmup_proportion),
                                 max_train_steps=all_step)
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
    optimizer = AdamWeightDecay(optimizer_parameters, learning_rate=lr_schedule, eps=args.adam_eps)
    return optimizer


def do_train(dataset=None, network=None, args=None):
    """do train"""
    optimizer = _create_optimizer(args, network, dataset.get_dataset_size())
    if args.model_path:
        param_dict = load_checkpoint(args.model_path)
        param_dict = load_state_dict(param_dict, args.model_config)
        load_param_into_net(network, param_dict)
    net_with_loss = LukeForReadingComprehensionWithLoss(network)
    config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1)
    ckpt_save_dir = args.output_dir
    if args.modelArts or args.duoka:
        ckpt_save_dir = os.path.join(ckpt_save_dir, 'ckpt_' + str(get_rank()))

    ckpoint_cb = ModelCheckpoint(prefix='luke_squad', directory=ckpt_save_dir, config=config_ck)
    netwithgrads = TrainOneStepCell(net_with_loss, optimizer)
    model = Model(netwithgrads)
    model.train(args.num_train_epochs, dataset, callbacks=[TimeMonitor(200), LossMonitor(25), ckpoint_cb],
                dataset_sink_mode=args.dataset_sink_mode)
