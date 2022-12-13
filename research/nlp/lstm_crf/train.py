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
#################train lstm-crf example on CoNLL2000########################
"""

import os
from copy import deepcopy
import numpy as np

from src import util
from src.util import get_chunks, get_label_lists, F1, LSTMCRFLearningRate
from src.model_utils.config import config
from src.dataset import get_data_set
from src.imdb import ImdbParser
from src.LSTM_CRF import Lstm_CRF

from mindspore.common import set_seed
from mindspore.nn.optim import AdamWeightDecay
from mindspore import Tensor, Model, context
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint

set_seed(1000)

def modelarts_pre_process():
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


def create_filter_fun(keywords):
    return lambda x: not (True in [key in x.name.lower() for key in keywords])


class EvalCabllBack(TimeMonitor):
    """print training log"""
    def __init__(self, network):

        self.parser = ImdbParser(config.data_CoNLL_path,
                                 config.glove_path,
                                 config.data_CoNLL_path,
                                 embed_size=config.embed_size)

        _, _, _, _, self.sequence_index, self.sequence_tag_index, self.tags_to_index_map \
            = self.parser.get_datas_embeddings(seg=['test'], build_data=config.build_data)

        self.ds_val = get_data_set(self.sequence_index, self.sequence_tag_index, config.batch_size)
        self.network = network
        self.weight = self.network.parameters_dict
        self.callback = F1(len(self.tags_to_index_map))
        self._best_val_F1 = 0

    def epoch_begin(self, run_context):
        self.network.is_training = True

    def epoch_end(self, run_context):
        """save .ckpt files"""
        self.network.is_training = False
        self.network.set_grad(False)
        self.model = Model(self.network)
        columns_list = ["feature", "label"]
        rest_golds_list = list()
        rest_preds_list = list()
        for data in self.ds_val.create_dict_iterator(num_epochs=1):
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            feature, label = input_data
            logits = self.model.predict_network(feature, label)
            logit_ids, label_ids = self.callback.update(logits, label)

            rest_preds = np.array(logit_ids)
            rest_preds = np.expand_dims(rest_preds, 0)

            rest_labels = deepcopy(label_ids)
            label_ids = np.expand_dims(label_ids, 0)
            rest_labels = np.expand_dims(rest_labels, 0)

            rest_golds, rest_preds = get_label_lists(rest_labels, rest_preds, label_ids)

            rest_golds_list += rest_golds
            rest_preds_list += rest_preds
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for golds, preds in zip(rest_golds_list, rest_preds_list):
            accs += [a == b for (a, b) in zip(golds, preds)]
            golds_chunks = set(get_chunks(golds, self.tags_to_index_map))
            preds_chunks = set(get_chunks(preds, self.tags_to_index_map))
            correct_preds += len(golds_chunks & preds_chunks)
            total_preds += len(preds_chunks)
            total_correct += len(golds_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        val_current_F1 = f1
        if self._best_val_F1 <= val_current_F1:
            self._best_val_F1 = val_current_F1
        print("current ACC {:.6f}%, current F1 {:.6f}%, self._best_val_F1 {:.6f}% "
              .format(acc*100, val_current_F1*100, self._best_val_F1*100))
        self.network.set_grad(True)


def train_lstm_crf():
    """ train lstm_crf """
    print('\ntrain.py config: \n', config)

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_id=config.device_id,
        enable_graph_kernel=False,
        device_target=config.device_target)

    parser = ImdbParser(config.data_CoNLL_path,
                        config.glove_path,
                        config.data_CoNLL_path,
                        embed_size=config.embed_size)
    # only create data
    if config.build_data:
        parser.build_datas(seg='train', build_data=config.build_data)
        return
    embeddings_size = config.embed_size
    embeddings, sequence_length, _, _, sequence_index, sequence_tag_index, tags_to_index_map \
        = parser.get_datas_embeddings(seg=['train'], build_data=config.build_data)

    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hiddle_size is multiples of 16, this problem will be solved later.
    embeddings_table = embeddings.astype(np.float32)
    if config.device_target == 'Ascend':
        pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
        if pad_num > 0:
            embeddings_table = np.pad(embeddings_table, [(0, 0), (0, pad_num)], 'constant')
        embeddings_size = int(np.ceil(config.embed_size / 16) * 16)
    ds_train = get_data_set(sequence_index, sequence_tag_index, config.batch_size)

    # create lstm_crf network
    network = Lstm_CRF(vocab_size=embeddings_table.shape[0],
                       tag_to_index=tags_to_index_map,
                       embedding_size=embeddings_size,
                       hidden_size=config.num_hiddens,
                       num_layers=config.num_layers,
                       weight=Tensor(embeddings_table),
                       bidirectional=config.bidirectional,
                       batch_size=config.batch_size,
                       seq_length=sequence_length,
                       dropout=config.dropout,
                       is_training=True)

    # create optimizer
    steps_per_epoch = ds_train.get_dataset_size()
    lr_schedule = LSTMCRFLearningRate(learning_rate=config.AdamWeightDecay.learning_rate,
                                      end_learning_rate=config.AdamWeightDecay.end_learning_rate,
                                      warmup_steps=int(steps_per_epoch * config.num_epochs * 0.02),
                                      decay_steps=steps_per_epoch * config.num_epochs,
                                      power=config.AdamWeightDecay.power)
    params = network.trainable_params()
    decay_params = list(filter(create_filter_fun(config.AdamWeightDecay.decay_filter), params))
    other_params = list(filter(lambda x: not create_filter_fun(config.AdamWeightDecay.decay_filter)(x), params))
    group_params = [{'params': decay_params, 'weight_decay': config.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0}]
    opt = AdamWeightDecay(params=group_params, learning_rate=lr_schedule, eps=config.AdamWeightDecay.eps)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**16, scale_factor=2, scale_window=100)
    if config.device_target == 'CPU':
        netwithgrads = util.Lstm_CRF_Cell_CPU(network, optimizer=opt, scale_update_cell=update_cell)
    else:
        netwithgrads = util.Lstm_CRF_Cell_Ascend(network, optimizer=opt, scale_update_cell=update_cell)
    model = Model(netwithgrads)

    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(network, param_dict)

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="lstm_crf", directory=config.ckpt_save_path, config=config_ck)
    eval_cb = EvalCabllBack(network)
    callbacks = [TimeMonitor(ds_train.get_dataset_size()),
                 util.LossCallBack(ds_train.get_dataset_size(), config.device_target),
                 eval_cb, ckpoint_cb]
    if config.device_target == "CPU":
        model.train(config.num_epochs, ds_train, callbacks=callbacks, dataset_sink_mode=False)
    else:
        model.train(config.num_epochs, ds_train, callbacks=callbacks)
    print("============== Training Success ==============")


if __name__ == '__main__':
    train_lstm_crf()
