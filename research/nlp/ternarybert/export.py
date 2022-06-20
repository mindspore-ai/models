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
# ===========================================================================

"""export checkpoint file into model"""

import argparse
import re
import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.config import student_net_cfg, task_cfg, cfg_cfg
from src.tinybert_model import BertModelCLS

parser = argparse.ArgumentParser(description="TernaryBert export model")
parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                    help="device where the code will be implemented. (Default: Ascend)")
parser.add_argument("--task_name", type=str, default="sts-b", choices=["sts-b", "QNLI", "SST-2"],
                    help="The name of the task to eval.")
parser.add_argument("--file_name", type=str, default="ternarybert", help="The name of the output file.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=["AIR", "MINDIR"],
                    help="output model type")
parser.add_argument("--ckpt_file", type=str, default="", help="pretrained checkpoint file")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

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


if __name__ == "__main__":
    task = Task(args.task_name)
    student_net_cfg.seq_length = task.seq_length
    student_net_cfg.batch_size = DEFAULT_BS
    student_net_cfg.do_quant = False

    eval_model = BertModelCLS(student_net_cfg, False, task.num_labels, 0.0, phase_type='student')
    param_dict = load_checkpoint(args.ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value
    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)
    input_ids = Tensor(np.zeros((student_net_cfg.batch_size, task.seq_length), np.int32))
    token_type_id = Tensor(np.zeros((student_net_cfg.batch_size, task.seq_length), np.int32))
    input_mask = Tensor(np.zeros((student_net_cfg.batch_size, task.seq_length), np.int32))

    input_data = [input_ids, token_type_id, input_mask]
    export(eval_model, *input_data, file_name=args.file_name, file_format=args.file_format)
