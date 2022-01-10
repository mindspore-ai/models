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

from src.util import F1, get_chunks, get_label_lists
from src.model_utils.config import config
from src.dataset import get_data_set
from src.LSTM_CRF import Lstm_CRF
from src.imdb import ImdbParser

from mindspore import Tensor, Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def modelarts_process():
    config.ckpt_file = os.path.join(config.output_path, config.ckpt_file)


def eval_lstm_crf():
    """ eval lstm """
    print('\neval.py config: \n', config)

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_id=config.device_id,
        device_target=config.device_target
        )

    embeddings_size = config.embed_size
    parser = ImdbParser(config.data_CoNLL_path,
                        config.glove_path,
                        config.data_CoNLL_path,
                        embed_size=config.embed_size
                        )
    embeddings, sequence_length, _, _, sequence_index, sequence_tag_index, tags_to_index_map \
        = parser.get_datas_embeddings(seg=['test'], build_data=False)
    embeddings_table = embeddings.astype(np.float32)

    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hiddle_size is multiples of 16, this problem will be solved later.
    if config.device_target == 'Ascend':
        pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
        if pad_num > 0:
            embeddings_table = np.pad(embeddings_table, [(0, 0), (0, pad_num)], 'constant')
        embeddings_size = int(np.ceil(config.embed_size / 16) * 16)
    ds_test = get_data_set(sequence_index, sequence_tag_index, config.batch_size)

    network = Lstm_CRF(vocab_size=embeddings.shape[0],
                       tag_to_index=tags_to_index_map,
                       embedding_size=embeddings_size,
                       hidden_size=config.num_hiddens,
                       num_layers=config.num_layers,
                       weight=Tensor(embeddings_table),
                       bidirectional=config.bidirectional,
                       batch_size=config.batch_size,
                       seq_length=sequence_length,
                       is_training=False)

    callback = F1(len(tags_to_index_map))
    model = Model(network)

    param_dict = load_checkpoint(os.path.join(config.ckpt_save_path, config.ckpt_path))
    load_param_into_net(network, param_dict)
    print("============== Starting Testing ==============")
    rest_golds_list = list()
    rest_preds_list = list()
    columns_list = ["feature", "label"]
    for data in ds_test.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        feature, label = input_data
        logits = model.predict(feature, label)
        logit_ids, label_ids = callback.update(logits, label)

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
        golds_chunks = set(get_chunks(golds, tags_to_index_map))
        preds_chunks = set(get_chunks(preds, tags_to_index_map))
        correct_preds += len(golds_chunks & preds_chunks)
        total_preds += len(preds_chunks)
        total_correct += len(golds_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)
    print("acc: {:.6f}%,  F1: {:.6f}% ".format(acc*100, f1*100))


if __name__ == '__main__':
    eval_lstm_crf()
