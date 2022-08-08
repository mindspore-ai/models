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

"""postprocess"""

import os
import argparse
import numpy as np
from mindspore import Tensor
from src.assessment_method import Accuracy, F1, Pearsonr, Matthews
from src.config import task_cfg, cfg_cfg


parser = argparse.ArgumentParser(description='postprocess')
parser.add_argument("--task_name", type=str, default="sts-b",
                    choices=['sts-b', "SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"],
                    help="The name of the task to train.")
parser.add_argument("--assessment_method", type=str, default="pearsonr",
                    choices=["accuracy", "f1", 'pearsonr', 'matthews'],
                    help="assessment_method include: [accuracy, bf1, mf1], default is accuracy")
parser.add_argument("--result_path", type=str, default="", help="result path")
parser.add_argument("--label_path", type=str, default="", help="label path")
args_opt = parser.parse_args()

DEFAULT_NUM_LABELS = cfg_cfg.DEFAULT_NUM_LABELS
DEFAULT_SEQ_LENGTH = cfg_cfg.DEFAULT_SEQ_LENGTH
DEFAULT_BS = cfg_cfg.DEFAULT_BS


class Task:
    """
    Encapsulation class of get the task parameter.
    """

    def __init__(self, task_name):
        self.task_name = task_name

    @property
    def num_labels(self):
        if self.task_name in task_cfg and "num_labels" in task_cfg[self.task_name]:
            return task_cfg[self.task_name]["num_labels"]
        return DEFAULT_NUM_LABELS

    @property
    def seq_length(self):
        if self.task_name in task_cfg and "seq_length" in task_cfg[self.task_name]:
            return task_cfg[self.task_name]["seq_length"]
        return DEFAULT_SEQ_LENGTH


task = Task(args_opt.task_name)


def eval_result_print(assessment_method="pearsonr", callback=None):
    """print eval result"""
    if assessment_method == "accuracy":
        print("accuracy is {}".format(callback.get_metrics()))
    elif assessment_method == "f1":
        print("F1 {:.6f} ".format(callback.get_metrics()))
    elif assessment_method == "matthews":
        print("matthews {:.6f} ".format(callback.get_metrics()))
    elif assessment_method == 'pearsonr':
        print("pearson {:.6f} ".format(callback.get_metrics()))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")

def get_acc():
    """
    calculate accuracy
    """
    if args_opt.assessment_method == "accuracy":
        callback = Accuracy()
    elif args_opt.assessment_method == "f1":
        callback = F1()
    elif args_opt.assessment_method == "matthews":
        callback = Matthews()
    elif args_opt.assessment_method == "pearsonr":
        callback = Pearsonr()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, matthews, pearsonr]")
    labels = np.load(args_opt.label_path)
    file_num = len(os.listdir(args_opt.result_path))
    for i in range(int(file_num/15)):
        f_name = "tinybert_bs" + str(DEFAULT_BS) + "_" + str(i) + "_13.bin"
        logits = np.fromfile(os.path.join(args_opt.result_path, f_name), np.float32)
        label_ids = labels[i]
        logits = logits.reshape(DEFAULT_BS, task.num_labels)
        callback.update(Tensor(logits), Tensor(label_ids))
    print("==============================================================")
    eval_result_print(args_opt.assessment_method, callback)
    print("==============================================================")


if __name__ == '__main__':
    get_acc()
