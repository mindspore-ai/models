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

"""task distill onnx script"""

import os
import mindspore.dataset as ds
from mindspore.common import set_seed, Tensor, dtype
from mindspore import log as logger
import onnxruntime as ort

from src.dataset import create_tinybert_dataset, DataType
from src.assessment_method import Accuracy, F1
from src.model_utils.config import config as args_opt, eval_cfg, td_teacher_net_cfg, td_student_net_cfg

_cur_dir = os.getcwd()
set_seed(123)
ds.config.set_seed(12345)

if args_opt.dataset_type == "tfrecord":
    dataset_type = DataType.TFRECORD
elif args_opt.dataset_type == "mindrecord":
    dataset_type = DataType.MINDRECORD
else:
    raise Exception("dataset format is not supported yet")


def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def eval_result_print(assessment_method="accuracy", callback=None):
    """print onnx eval result"""
    if assessment_method == "accuracy":
        print("============== acc is {}".format(callback.acc_num / callback.total_num))
    elif assessment_method == "bf1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mf1":
        print("F1 {:.6f} ".format(callback.eval()))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")


def do_eval_standalone():
    """
    do onnx eval standalone
    """
    session, [ids, ids2, mask] = create_session(args_opt.onnx_path, args_opt.device_target)
    eval_dataset = create_tinybert_dataset('td', batch_size=eval_cfg.batch_size,
                                           device_num=1, rank=0, do_shuffle="false",
                                           data_dir=args_opt.eval_data_dir,
                                           schema_dir=args_opt.schema_dir,
                                           data_type=dataset_type)
    print('eval dataset size: ', eval_dataset.get_dataset_size())
    print('eval dataset batch size: ', eval_dataset.get_batch_size())
    if args_opt.assessment_method == "accuracy":
        callback = Accuracy()
    elif args_opt.assessment_method == "bf1":
        callback = F1(num_labels=args_opt.num_labels)
    elif args_opt.assessment_method == "mf1":
        callback = F1(num_labels=args_opt.num_labels, mode="MultiLabel")
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        [logits] = session.run(None, {ids: input_ids, ids2: token_type_id, mask: input_mask})
        logits = Tensor(logits, dtype.float32)
        label_ids = Tensor(label_ids, dtype.float32)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(args_opt.assessment_method, callback)
    print("==============================================================")


def run_main():
    """task_distill function"""
    if args_opt.do_train.lower() != "true" and args_opt.do_eval.lower() != "true":
        raise ValueError("do train or do eval must have one be true, please confirm your config")
    if args_opt.task_name in ["SST-2", "QNLI", "MNLI", "TNEWS"] and args_opt.task_type != "classification":
        raise ValueError(f"{args_opt.task_name} is a classification dataset, please set --task_type=classification")
    if args_opt.task_name in ["CLUENER"] and args_opt.task_type != "ner":
        raise ValueError(f"{args_opt.task_name} is a ner dataset, please set --task_type=ner")
    if args_opt.task_name in ["SST-2", "QNLI", "MNLI"] and \
            (td_teacher_net_cfg.vocab_size != 30522 or td_student_net_cfg.vocab_size != 30522):
        logger.warning(f"{args_opt.task_name} is an English dataset. Usually, we use 21128 for CN vocabs and 30522 for "
                       f"EN vocabs according to the origin paper.")
    if args_opt.task_name in ["TNEWS", "CLUENER"] and \
            (td_teacher_net_cfg.vocab_size != 21128 or td_student_net_cfg.vocab_size != 21128):
        logger.warning(f"{args_opt.task_name} is a Chinese dataset. Usually, we use 21128 for CN vocabs and 30522 for "
                       f"EN vocabs according to the origin paper.")

    do_eval_standalone()


if __name__ == '__main__':
    run_main()
