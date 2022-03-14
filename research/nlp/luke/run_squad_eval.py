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
"""run squad eval"""
import collections
import os

from tqdm import tqdm

from src.utils.model_utils import ModelArchive
from src.utils.word_tokenizer import AutoTokenizer
from src.luke.config import LukeConfig
from src.reading_comprehension.dataLoader import load_eval
from src.reading_comprehension.model import LukeForReadingComprehension
from src.reading_comprehension.squad_get_predictions import write_predictions
from src.reading_comprehension.squad_postprocess import SQuad_postprocess
from src.model_utils.config_args import args_config as args

from mindspore import context, load_checkpoint, load_param_into_net, Model

args.data_dir = os.path.join(args.data, 'squad')
context.set_context(mode=context.GRAPH_MODE, device_target=args.device)

# load pretrain
model_archive = ModelArchive.load(args.model_file)
args.bert_model_name = model_archive.bert_model_name
args.max_mention_length = model_archive.max_mention_length
args.model_path = model_archive.model_path
args.tokenizer = AutoTokenizer.from_pretrained(model_archive.bert_model_name)
luke_config = LukeConfig(**model_archive.metadata["model_config"])
args.model_config = luke_config


# do eval
def do_eval():
    """do eval"""
    # model art
    network = LukeForReadingComprehension(luke_config)
    checkpoint_file = args.checkpoint_file
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(network, param_dict)
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    dataloader, examples, features, processor = load_eval(args)
    all_results = []
    model = Model(network)
    for batch in tqdm(dataloader.create_dict_iterator(), desc="eval"):
        inputs = {k: v for k, v in batch.items() if k != "example_indices"}
        column_list = []
        for d in inputs:
            column_list.append(inputs[d])
        word_ids, word_segment_ids, word_attention_mask, entity_ids, \
        entity_position_ids, entity_segment_ids, entity_attention_mask = column_list
        outputs = model.predict(word_ids, word_segment_ids, word_attention_mask,
                                entity_ids, entity_position_ids, entity_segment_ids,
                                entity_attention_mask)
        for i, example_index in enumerate(batch["example_indices"]):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits, end_logits = [o[i] for o in outputs]
            start_logits = start_logits.asnumpy()
            end_logits = end_logits.asnumpy()
            all_results.append(RawResult(unique_id, start_logits, end_logits))
    all_predictions = write_predictions(args, examples, features, all_results, 20, 30, False)
    SQuad_postprocess(os.path.join(args.data_dir, processor.dev_file), all_predictions, output_metrics="output.json")

    print("over")


do_eval()
