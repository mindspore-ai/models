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

import os
import collections
import argparse
import pickle
import numpy as np
from mindspore import Tensor
from src.assessment_method import Accuracy


def classifier_eval(cb, nc, assessment):
    """task classifier eval"""
    file_name = os.listdir(args.label_dir)
    for f in file_name:
        f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        logits = np.fromfile(f_name, np.float32).reshape(args.batch_size, nc)
        logits = Tensor(logits)
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, 1))
        cb.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(assessment, cb)
    print("==============================================================")


def squad_eval(nc, seq_len):
    """task squad eval"""
    from src.squad_utils import read_squad_examples
    from src.squad_get_predictions import get_result
    from src.squad_postprocess import SQuad_postprocess

    with open(args.eval_data_file_path, "rb") as fin:
        eval_features = pickle.load(fin)

    eval_examples = read_squad_examples(args.eval_json_path, False)

    file_name = os.listdir(args.label_dir)
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_log_prob", "end_log_prob"])
    outputs = []
    for f in file_name:
        logit_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        unique_id_name = os.path.join(args.label_dir, f)
        mask_name = os.path.join(args.input1_path, f)

        logits = np.fromfile(logit_name, np.float32)
        logits = logits.reshape(args.batch_size, seq_len, nc)
        input_mask = np.fromfile(mask_name, np.float32).reshape(args.batch_size, seq_len)
        unique_id = np.fromfile(unique_id_name, np.int32)

        start_logits = np.squeeze(logits[:, :, 0:1], axis=-1)
        start_logits = start_logits + 100 * input_mask
        end_logits = np.squeeze(logits[:, :, 1:2], axis=-1)
        end_logits = end_logits + 100 * input_mask

        for i in range(args.batch_size):
            unique_id = int(unique_id[i])
            start_logits = [float(x) for x in start_logits[i].flat]
            end_logits = [float(x) for x in end_logits[i].flat]
            outputs.append(RawResult(
                unique_id=unique_id,
                start_log_prob=start_logits,
                end_log_prob=end_logits))

    all_predictions, _ = get_result(outputs, eval_examples, eval_features)
    SQuad_postprocess(args.eval_json_path, all_predictions, output_metrics="output.json")


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    f1 = 0.0
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
        f1 = round(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN), 6)
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")
    return f1


parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
parser.add_argument("--input1_path", type=str, default="./preprocess_Result/01_data", help="input mask path")
parser.add_argument("--task_type", type=str, default='sst2', help="dataset name")
parser.add_argument("--eval_data_file_path", type=str, default="",
                    help="Data path, it is better to use absolute path")
parser.add_argument('--eval_json_path', type=str, default="", help='eval json path')

args = parser.parse_args()

if __name__ == "__main__":
    args.batch_size = 1
    assessment_type = "accuracy"
    call_back = Accuracy()
    if args.task_type == 'mnli':
        num_class = 3
        classifier_eval(call_back, num_class, assessment=assessment_type)
    elif args.task_type == 'sst2':
        num_class = 2
        classifier_eval(call_back, num_class, assessment=assessment_type)
    elif args.task_type == 'squadv1':
        classes = 2
        seq_length = 384
        if args.eval_json_path:
            squad_eval(classes, seq_length)
        else:
            raise ValueError("eval json path must be provided when task type is squadv1")
    else:
        raise ValueError("dataset not supported, support: [mnli, sst2, squadv1]")
