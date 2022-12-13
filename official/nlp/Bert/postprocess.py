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
postprocess script.
'''

import argparse
import collections
import os
import pickle
from glob import glob

import numpy as np
from mindspore import Tensor

from src import tokenization
from src.assessment_method import F1, MCC, Accuracy, Spearman_Correlation
from src.create_squad_data import convert_examples_to_features, read_squad_examples
from src.squad_get_predictions import write_predictions
from src.squad_postprocess import SQuad_postprocess


def eval_result_print(assessment_method_="accuracy", callback_=None):
    """print eval result"""
    if assessment_method_ == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback_.acc_num, callback_.total_num,
                                                                  callback_.acc_num / callback_.total_num))
    elif assessment_method_ == "bf1":
        print("Precision {:.6f} ".format(callback_.TP / (callback_.TP + callback_.FP)))
        print("Recall {:.6f} ".format(callback_.TP / (callback_.TP + callback_.FN)))
        print("F1 {:.6f} ".format(2 * callback_.TP / (2 * callback_.TP + callback_.FP + callback_.FN)))
    elif assessment_method_ == "mf1":
        print("F1 {:.6f} ".format(callback_.eval()[0]))
    elif assessment_method_ == "mcc":
        print("MCC {:.6f} ".format(callback_.cal()))
    elif assessment_method_ == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback_.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

def eval_squad_result(opts):
    tokenizer = tokenization.FullTokenizer(vocab_file=opts.vocab_file_path, do_lower_case=True)

    if not os.path.exists('eval_features.bin') or not os.path.exists('eval_examples.bin'):
        print("Now convert examples to features.....")
        eval_examples = read_squad_examples(opts.eval_json_path, False)
        with open('eval_examples.bin', 'wb') as fp:
            pickle.dump(eval_examples, fp, protocol=1)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=opts.seq_length,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=None,
            vocab_file=opts.vocab_file_path)
        with open('eval_features.bin', 'wb') as fp:
            pickle.dump(eval_features, fp, protocol=1)
    else:
        print("Features found, loading...")
        with open('eval_examples.bin', 'rb') as fp:
            eval_examples = pickle.load(fp)
        with open('eval_features.bin', 'rb') as fp:
            eval_features = pickle.load(fp)

    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    res_files = glob(opts.result_dir+'/*.bin')
    res_files.sort()
    mask_path = os.path.join(opts.preprocess_path, "01_data")
    unique_path = os.path.join(opts.preprocess_path, "03_data")
    outputs = []
    for res in res_files:
        logit = np.fromfile(res, np.float32).reshape(opts.batch_size, opts.seq_length, 2)
        res_name = os.path.basename(res)
        idx = res_name.split('_')[2]

        unique_ids = np.fromfile(f'{unique_path}/squad_bs{opts.batch_size}_{idx}.bin', np.int32)
        input_mask = np.fromfile(f'{mask_path}/squad_bs{opts.batch_size}_{idx}.bin', np.int32)
        input_mask = input_mask.reshape(opts.batch_size, opts.seq_length)

        for i in range(opts.batch_size):
            start_logits = logit[i, :, 0]
            start_logits = start_logits + 100 * input_mask[i]
            start_logits = list(start_logits)

            end_logits = logits[i, :, 1]
            end_logits = end_logits + 100 * input_mask[i]
            end_logits = list(end_logits)

            outputs.append(RawResult(
                unique_id=unique_ids[i],
                start_logits=start_logits,
                end_logits=end_logits))

    all_predictions = write_predictions(eval_examples, eval_features, outputs, 20, 30, True)
    SQuad_postprocess(opts.eval_json_path, all_predictions, output_metrics="output.json")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="seq_length, default is 128. You can get this value through the relevant'*.yaml' filer")
    parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--label_dir", type=str, default="", help="label data dir")
    parser.add_argument("--assessment_method", type=str, default="BF1",
                        choices=["BF1", "clue_benchmark", "MF1", "Accuracy"],
                        help="assessment_method include: [BF1, clue_benchmark, MF1], default is BF1")
    parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
    parser.add_argument("--task", type=str, default="ner", choices=["ner", "ner_crf", "classifier", "squad"],
                        help="task, include: [ner, ner_crf, classifier, squad], default is ner")
    parser.add_argument("--preprocess_path", type=str, default="preprocess_Result", help="path to preprocess Files")
    parser.add_argument("--vocab_file_path", type=str, default="/path/to/vocab_bert_large_en.txt", help="vocab file")
    parser.add_argument("--eval_json_path", type=str, default="/path/to/dev-v1.1.json", help="eval json")

    args, _ = parser.parse_known_args()

    task = args.task.lower()
    if args.task == 'squad':
        eval_squad_result(args)
        exit()
    if task == "classifier":
        num_class = 15
    else:
        num_class = 41

    assessment_method = args.assessment_method.lower()
    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "bf1":
        callback = F1((task == "ner_crf"), num_class)
    elif assessment_method == "mf1":
        callback = F1((task == "ner_crf"), num_labels=num_class, mode="MultiLabel")
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = Spearman_Correlation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    file_name = os.listdir(args.label_dir)
    for f in file_name:
        if task == "ner_crf":
            logits = ()
            for j in range(args.seq_length):
                f_name = f.split('.')[0] + '_' + str(j) + '.bin'
                data_tmp = np.fromfile(os.path.join(args.result_dir, f_name), np.int32)
                data_tmp = data_tmp.reshape(args.batch_size, num_class + 2)
                logits += ((Tensor(data_tmp),),)
            f_name = f.split('.')[0] + '_' + str(args.seq_length) + '.bin'
            data_tmp = np.fromfile(os.path.join(args.result_dir, f_name), np.int32).tolist()
            data_tmp = Tensor(data_tmp)
            logits = (logits, data_tmp)
        else:
            f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
            logits = np.fromfile(f_name, np.float32).reshape(args.seq_length * args.batch_size, num_class)
            logits = Tensor(logits)
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, args.seq_length))
        callback.update(logits, label_ids)

    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")
