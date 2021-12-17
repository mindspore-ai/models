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
"""define savecallback, save best model while training."""
import time
from mindspore.train.callback import Callback
from .inference import (Inference, load_ground_truth, load_result,
                        remove_nonexistent_ground_truth, calculate_clip_acc)


class SaveCallback(Callback):
    """
    define savecallback, save best model while training.
    """

    def __init__(self, model, eval_dataset_save, epoch_per_eval, cfg):
        super(SaveCallback, self).__init__()
        self.model = model
        self.inference = Inference()
        self.inference_dataset = eval_dataset_save
        self.acc = 0.85
        self.save_path = cfg.result_path
        self.epoch_per_eval = epoch_per_eval
        self.cfg = cfg

    def epoch_end(self, run_context):
        """
        eval and save model while training.
        """
        t1 = time.time()
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch % self.epoch_per_eval == 0:
            print("\n=======================Inference====================\n")

            inference_results, clip_inference_results = self.inference(self.inference_dataset, self.model,
                                                                       self.cfg.annotation_path)
            print('load ground truth')
            ground_truth, class_labels_map = load_ground_truth(
                self.cfg.annotation_path, "validation")
            print('number of ground truth: {}'.format(len(ground_truth)))

            n_ground_truth_top_1 = len(ground_truth)
            n_ground_truth_top_5 = len(ground_truth)

            result_top1, result_top5 = load_result(
                clip_inference_results, class_labels_map)

            ground_truth_top1 = remove_nonexistent_ground_truth(
                ground_truth, result_top1)
            ground_truth_top5 = remove_nonexistent_ground_truth(
                ground_truth, result_top5)

            if self.cfg.ignore:
                n_ground_truth_top_1 = len(ground_truth_top1)
                n_ground_truth_top_5 = len(ground_truth_top5)

            correct_top1 = [1 if line[1] in result_top1[line[0]]
                            else 0 for line in ground_truth_top1]
            correct_top5 = [1 if line[1] in result_top5[line[0]]
                            else 0 for line in ground_truth_top5]

            clip_acc = calculate_clip_acc(
                inference_results, ground_truth, class_labels_map)
            accuracy_top1 = float(sum(correct_top1)) / \
                float(n_ground_truth_top_1)
            accuracy_top5 = float(sum(correct_top5)) / \
                float(n_ground_truth_top_5)
            print('==================Accuracy=================\n'
                  ' clip-acc : {} \ttop-1 : {} \ttop-5: {}'.format(clip_acc, accuracy_top1, accuracy_top5))
            t2 = time.time()

            print("Eval in training Time consume: ", t2 - t1, "\n")
