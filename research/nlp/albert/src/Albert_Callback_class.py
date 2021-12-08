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

"""This is callback program"""

import os
from mindspore import Model
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from src.albert_for_finetune import AlbertCLS
from src.model_utils.config import albert_net_cfg
from src.assessment_method import Accuracy, F1, MCC, Spearman_Correlation


class albert_callback(Callback):
    """Classifier task callback"""

    def __init__(self, net, args_opt, steps_per_epoch, ds_eval, save_checkpoint_path):
        self.net = net
        self.best_output = 0
        self.best_epoch = 0
        self.args_opt = args_opt
        self.steps_per_epoch = steps_per_epoch
        self.ds_eval = ds_eval
        self.path_url = self.args_opt.output_path
        self.save_checkpoint_path = save_checkpoint_path

    def epoch_end(self, run_context):
        """epoch end"""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num

        finetune_checkpoint_root = self.args_opt.load_finetune_checkpoint_path
        finetune_checkpoint_path = os.path.join(self.save_checkpoint_path, 'classifier-' + str(cur_epoch) + '_'
                                                + str(self.steps_per_epoch) + '.ckpt')

        output = do_eval(self.ds_eval, AlbertCLS, self.args_opt.num_class, self.args_opt.assessment_method.lower(),
                         finetune_checkpoint_path)
        if self.args_opt.device_num == 1 or (self.args_opt.device_id == 0 and self.args_opt.device_num == 8):
            if output > self.best_output:
                if not self.args_opt.enable_modelarts:
                    self.path_url = finetune_checkpoint_root
                self.best_output = output
                self.best_epoch = cur_epoch
                best_file_name = 'best_{}_{:.5f}.ckpt'.format(self.args_opt.assessment_method.lower(), self.best_output)
                save_checkpoint(self.net, os.path.join(self.path_url, best_file_name))
                if self.args_opt.enable_modelarts:
                    import moxing as mox
                    mox.file.copy_parallel(src_url=self.path_url,
                                           dst_url=self.args_opt.train_url)

                log_text = 'EPOCH: {:d}, {}: {:.5f}'.format(cur_epoch, self.args_opt.assessment_method.lower(), output)
                print(log_text)
                log_text = 'BEST {}: {:0.5f}, BEST EPOCH: {}'.format(self.args_opt.assessment_method.lower(),
                                                                     self.best_output,
                                                                     self.best_epoch)
                print(log_text)


def do_eval(dataset=None, network=None, num_class=2, assessment_method="accuracy", load_checkpoint_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(albert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(False, num_class)
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = Spearman_Correlation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        callback.update(logits, label_ids)
    print("==============================================================")
    result = eval_result_print(assessment_method, callback)
    print("==============================================================")
    return result


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
        precision = round(callback.acc_num / callback.total_num, 6)
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
        precision = round(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN), 6)
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
        precision = round(callback.cal(), 2)
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
        precision = round(callback.cal()[0], 2)
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")
    return precision
