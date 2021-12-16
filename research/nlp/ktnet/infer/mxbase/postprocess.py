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
postprocess script.
"""

import argparse
import collections
import glob
import os
import numpy as np

from src.reader.squad_twomemory import DataProcessor as SquadDataProcessor
from src.reader.squad_twomemory import write_predictions as write_predictions_squad

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--result_dir", type=str, default="",
                        help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
    parser.add_argument("--checkpoints", type=str, default="./squad", help="Path to save checkpoints")
    parser.add_argument("--data_url", type=str, default="../data/rawdata", help="Path to save data")
    parser.add_argument("--label_dir", type=str, default="../data/input", help="Path to save label")
    args_opt = parser.parse_args()
    return args_opt


def get_infer_logits(args, file_name):
    """
    get the result of model output.
    Args:
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 384.
    """
    infer_logits_path = os.path.realpath(os.path.join(args.result_dir, "result", file_name))
    logits = []
    with open(infer_logits_path, "r") as f:
        for line in f:
            logits.append(float(line.strip('\n')))

    logits = np.array(logits).reshape((2, 1, 384))
    start_logits, end_logits = np.split(logits, 2, 0)

    return start_logits, end_logits


def read_concept_embedding(embedding_path):
    """read concept embedding"""
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    return concept2id


def run():
    """
    read pipeline and do infer
    """
    args = parse_args()
    # input_ids file list, every file content a tensor[1,128]
    file_list = glob.glob(os.path.join(os.path.realpath(args.result_dir), "result", "*.txt"))
    cwq_lists = []
    for i in range(len(file_list)):
        b = os.path.split(file_list[i])
        cwq_lists.append(b)

    yms_lists = []
    for i in range(len(cwq_lists)):
        c = cwq_lists[i][0] + '/' + cwq_lists[i][1]
        yms_lists.append(c)
    file_list = yms_lists

    all_results = []
    for input_ids in file_list:
        file_name = input_ids.split('/')[-1].split('.')[0] + '.bin'
        start_logits, end_logits = get_infer_logits(args, input_ids.split('/')[-1])

        label_file = os.path.realpath(os.path.join(args.label_dir, "06_data", file_name))
        unique_ids = np.fromfile(label_file, np.int64)

        np_unique_ids = unique_ids[0].reshape(1, 1)
        np_start_logits = np.squeeze(start_logits, axis=0)
        np_end_logits = np.squeeze(end_logits, axis=0)

        for idx in range(np_unique_ids.shape[0]):
            if len(all_results) % 1000 == 0:
                print("Processing example: %d" % len(all_results))
            unique_id = int(np_unique_ids[idx])
            start_logits = [float(x) for x in np_start_logits[idx].flat]
            end_logits = [float(x) for x in np_end_logits[idx].flat]

            all_results.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    output_prediction_file = os.path.join(args.checkpoints, "predictions.json")
    output_nbest_file = os.path.join(args.checkpoints, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.checkpoints, "null_odds.json")
    output_evaluation_result_file = os.path.join(args.checkpoints, "eval_result.json")

    wn_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/wn_concept2vec.txt")
    nell_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/nell_concept2vec.txt")

    processor = SquadDataProcessor(
        vocab_path=args.data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
        do_lower_case=False,
        max_seq_length=384,
        in_tokens=False,
        doc_stride=128,
        max_query_length=64)

    eval_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_squad/tokens/dev.tokenization.cased.data',
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path': args.data_url + "/retrieve_wordnet/output_squad/retrived_synsets.data",
        'use_nell': True,
        'retrieved_nell_concept_path': args.data_url + "/retrieve_nell/output_squad/dev.retrieved_nell_concepts.data",
    }
    processor.data_generator(
        data_path=args.data_url + "/SQuAD/dev-v1.1.json",
        batch_size=1,
        phase='predict',
        shuffle=False,
        dev_count=1,
        epoch=1,
        **eval_concept_settings)

    features = processor.get_features(
        processor.predict_examples, is_training=False, **eval_concept_settings)
    eval_result = write_predictions_squad(processor.predict_examples, features, all_results,
                                          20, 30, False, output_prediction_file,
                                          output_nbest_file, output_null_log_odds_file,
                                          False, 0.0, False, args.data_url + '/SQuAD/dev-v1.1.json',
                                          output_evaluation_result_file)

    print("==============================================================")
    print(eval_result)
    print("==============================================================")

if __name__ == '__main__':
    run()
