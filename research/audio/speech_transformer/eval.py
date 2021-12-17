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
"""Transformer evaluation script."""

import json
import os

import numpy as np
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from evaluate_cer import evaluate_cer
from src.dataset import MsAudioDataset
from src.dataset import create_transformer_dataset
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.transformer_model import TransformerModel


config.dtype = mstype.float32
config.compute_type = mstype.float16
config.batch_size = config.batch_size_ev
config.hidden_dropout_prob = config.hidden_dropout_prob_ev
config.attention_probs_dropout_prob = config.attention_probs_dropout_prob_ev


class TransformerInferCell(nn.Cell):
    """
    Encapsulation class of transformer network infer.
    """
    def __init__(self, network):
        super(TransformerInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, source_ids, source_mask):
        predicted_ids = self.network(source_ids, source_mask)
        return predicted_ids


def load_weights(model_path):
    """
    Load checkpoint as parameter dict, support both npz file and mindspore checkpoint file.
    """
    if model_path.endswith(".npz"):
        ms_ckpt = np.load(model_path)
        is_npz = True
    else:
        ms_ckpt = load_checkpoint(model_path)
        is_npz = False

    weights = {}
    for msname in ms_ckpt:
        infer_name = msname
        if "tfm_decoder" in msname:
            infer_name = "tfm_decoder.decoder." + infer_name
        if is_npz:
            weights[infer_name] = ms_ckpt[msname]
        else:
            weights[infer_name] = ms_ckpt[msname].data.asnumpy()
    weights["tfm_decoder.decoder.tfm_embedding_lookup.embedding_table"] = \
        weights["tfm_embedding_lookup.embedding_table"]

    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name]), name=name)
    return parameter_dict


def modelarts_pre_process():
    """modelarts pre process"""
    config.output_file = os.path.join(config.output_path, config.output_file)
    config.data_file = os.path.join(config.data_file, config.data_file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_transformer_eval():
    """
    Transformer evaluation.
    """
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        reserve_class_name_in_scope=False,
        device_id=get_device_id(),
    )

    dataset = create_transformer_dataset(
        epoch_count=1,
        rank_size=1,
        rank_id=0,
        do_shuffle='false',
        data_json_path=config.data_json_path,
        chars_dict_path=config.chars_dict_path,
        batch_size=config.batch_size_ev,
    )
    char_list, _, _ = MsAudioDataset.process_dict(config.chars_dict_path)
    tfm_model = TransformerModel(config=config, is_training=False, use_one_hot_embeddings=False)

    parameter_dict = load_weights(config.model_file)
    load_param_into_net(tfm_model, parameter_dict)

    tfm_infer = TransformerInferCell(tfm_model)
    model = Model(tfm_infer)

    predictions = []
    target_sents = []
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        target_sents.append(batch["target_eos_ids"])
        source_feats = Tensor(batch["source_eos_features"], mstype.float32)
        source_mask = Tensor(batch["source_eos_mask"], mstype.int32)
        predicted_ids = model.predict(source_feats, source_mask)
        predictions.append(predicted_ids.asnumpy())

    result_dict = dict()
    sample_num = 0
    for batch_out, batch_gt in zip(predictions, target_sents):
        for i in range(config.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            predicted_tokens = [char_list[x] for x in batch_out[i].tolist()]
            predict = " ".join(predicted_tokens)
            gt_tokens = [char_list[x] for x in batch_gt[i].tolist() if x != -1]
            gt = " ".join(gt_tokens)
            result_dict[sample_num] = {
                'output': predict,
                'gt': gt,
            }
            sample_num += 1

    with open(config.output_file, 'w') as file:
        json.dump(result_dict, file, indent=2)


if __name__ == "__main__":
    run_transformer_eval()
    evaluate_cer()
