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
# ===========================================================================
"""DAM EvalMetric"""
from mindspore.nn.metrics import Metric
from src import ubuntu_evaluation as ub_eval
from src import douban_evaluation as db_eval


class EvalMetric(Metric):
    """DAM EvalMetric"""

    def __init__(self, model_name="DAM_ubuntu", score_file=None):
        super(EvalMetric, self).__init__()
        self.model_name = model_name
        self.score_file = score_file
        self.pred_probs = []
        self.true_labels = []
        self.global_step = 0

    def clear(self):
        """Clear the internal evaluation result."""
        self.pred_probs = []
        self.true_labels = []
        self.global_step = 0

    def update(self, *inputs):
        """Update list of predicts and labels."""
        self.global_step += 1
        print('eval {} its'.format(self.global_step))
        batch_predict = inputs[0].asnumpy()
        batch_label = inputs[1].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())

    def eval(self):
        """Evaluating"""
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size() is not equal to pred_probs.size()')
        if self.model_name == "DAM_ubuntu":
            auc = ub_eval.evaluate_m(self.pred_probs, self.true_labels)
        elif self.model_name == "DAM_douban":
            auc = db_eval.evaluate_m(self.pred_probs, self.true_labels)
        else:
            raise RuntimeError('Evaluation function is not defined')

        if self.score_file is not None:
            with open(self.score_file, 'w') as file_out:
                for i in range(len(self.true_labels)):
                    file_out.write(str(self.pred_probs[i]) + '\t' + str(self.true_labels[i]) + '\n')

        return auc
