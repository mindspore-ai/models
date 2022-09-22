# Copyright 2020 Huawei Technologies Co., Ltd
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
""" evaluation """

import os
import json
import mindspore.nn as nn
from mindspore import set_seed
from mindspore import context, Tensor, load_param_into_net, Model, load_checkpoint, Parameter
from mindspore.context import ParallelMode
import mindspore.common.dtype as mstype
from mindspore.communication import init, get_group_size, get_rank
from src import tokenization
from src.dataset import create_dataset
from src.model_encoder_decoder import EncoderDecoderConfig
from src.model_infer import EncoderDecoderInferModel
from src.model_utils.moxing_adapter import get_device_num
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.process_output import process_output, get_target
from src.roberta_model import RobertaGenerationConfig
from src.rouge_score import get_rouge_score



config.dtype = mstype.float32
config.compute_type = mstype.float16

set_seed(2022)


class EncoderDecoderInferCell(nn.Cell):
    """
    Encapsulation class of Roberta_Seq2Seq network infer.
    """

    def __init__(self, network):
        super(EncoderDecoderInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, input_ids, attention_mask):
        predicted_ids = self.network(input_ids, attention_mask)
        return predicted_ids


def load_weights(model_path):
    """
    Load checkpoint as parameter dict, support both npz file and mindspore checkpoint file.
    """

    ms_ckpt = load_checkpoint(model_path)

    weights = {}
    for msname in ms_ckpt:
        infer_name = msname
        weights[infer_name] = ms_ckpt[msname].data.asnumpy()
        if 'decoder' in infer_name:
            new_infer_name = 'beam_decoder.' + infer_name
            weights[new_infer_name] = ms_ckpt[infer_name].data.asnumpy()
    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name]), name=name)
    return parameter_dict


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.test_data_dir = os.path.join(config.data_path, 'test.mindrecord')
    config.output_file = os.path.join(config.output_path, 'eval_result.txt')
    config.model_file = os.path.join(
        config.load_path, 'roberta_seq2seq_last.ckpt')
    config.vocab_file_path = os.path.join(config.load_path, 'vocab.json')
    config.test_data_json_dir = os.path.join(config.data_path, 'test.json')


def get_config():
    config.test_data_dir = os.path.join(config.data_path, 'test.mindrecord')
    config.test_data_json_dir = os.path.join(config.data_path, 'test.json')


# @moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """
    Transformer evaluation.
    """
    # global config
    if not config.enable_modelarts:
        get_config()
    cfg = config
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    cfg.device_num = get_device_num()

    if cfg.device_target == "Ascend":
        device_id = get_device_id()
        if cfg.device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=cfg.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        else:
            context.set_context(device_id=device_id)

    config.use_cache = True
    encoder_config = RobertaGenerationConfig(config)
    decoder_config = RobertaGenerationConfig(config, is_decoder=True, add_cross_attention=True)

    model_config = EncoderDecoderConfig(encoder_config, decoder_config, beam_width=cfg.beam_width,
                                        length_penalty_weight=cfg.length_penalty_weight,
                                        max_decode_length=cfg.max_decode_length, batch_size=cfg.batch_size)

    ende_model = EncoderDecoderInferModel(
        config=model_config, is_training=False, add_pooling_layer=False)

    parameter_dict = load_weights(cfg.model_file)
    load_param_into_net(ende_model, parameter_dict)

    ende_infer = EncoderDecoderInferCell(ende_model)
    model = Model(ende_infer)
    rank_size, rank_id = _get_rank_info()

    test_data = create_dataset(
        cfg.batch_size, data_file_path=cfg.test_data_dir, do_shuffle=False, rank_size=rank_size,
        rank_id=rank_id)

    predictions = []
    source_sents = []
    target_sents = []
    for batch in test_data.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["input_ids"])
        target_sents.append(batch["decoder_input_ids"])
        source_ids = Tensor(batch["input_ids"], mstype.int32)
        source_mask = Tensor(batch["attention_mask"], mstype.int32)
        predicted_ids = model.predict(source_ids, source_mask)
        predictions.append(predicted_ids.asnumpy())
    f = open(cfg.output_file, 'w')
    result = []
    for batch_out in predictions:
        for i in range(cfg.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            token_ids = [str(x) for x in batch_out[i].tolist()]
            result.append(token_ids)
    json.dump(result, f, ensure_ascii=False)
    f.close()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=cfg.vocab_file_path, do_lower_case=False)

    pred = process_output(predictions, tokenizer, skip_special_tokens=True)
    target = get_target(file=cfg.test_data_json_dir, num_samples=len(pred))
    # rouge(pred, target)
    rouge_output = get_rouge_score(
        predictions=pred, references=target, rouge_types=["rouge1", "rouge2", "rougeL"])
    rouge1 = rouge_output['rouge1']
    rouge2 = rouge_output['rouge2']
    rougeL = rouge_output['rougeL']
    rouge1_dict = {'P': [], 'R': [], 'F': []}
    rouge2_dict = {'P': [], 'R': [], 'F': []}
    rougeL_dict = {'P': [], 'R': [], 'F': []}
    for t in rouge1:
        rouge1_dict['P'].append(t.precision)
        rouge1_dict['R'].append(t.recall)
        rouge1_dict['F'].append(t.fmeasure)
    for t in rouge2:
        rouge2_dict['P'].append(t.precision)
        rouge2_dict['R'].append(t.recall)
        rouge2_dict['F'].append(t.fmeasure)
    for t in rougeL:
        rougeL_dict['P'].append(t.precision)
        rougeL_dict['R'].append(t.recall)
        rougeL_dict['F'].append(t.fmeasure)

    rouge1_mean = {k: float(sum(values)) / len(values)
                   for k, values in rouge1_dict.items()}
    rouge2_mean = {k: float(sum(values)) / len(values)
                   for k, values in rouge2_dict.items()}
    rougeL_mean = {k: float(sum(values)) / len(values)
                   for k, values in rougeL_dict.items()}
    print(rouge1_mean)
    print(rouge2_mean)
    print(rougeL_mean)


if __name__ == '__main__':
    run_eval()
    print('over')
