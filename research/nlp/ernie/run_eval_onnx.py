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
'''
ERNIE task classifier onnx script.
'''

import os
import argparse
from src.dataset import create_finetune_dataset
from src.assessment_method import Accuracy, F1
from mindspore import Tensor, dtype
import onnxruntime as ort


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")

def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name_0 = session.get_inputs()[0].name
    input_name_1 = session.get_inputs()[1].name
    input_name_2 = session.get_inputs()[2].name
    output_name_0 = session.get_outputs()[0].name
    return session, input_name_0, input_name_1, input_name_2, output_name_0


def do_eval_onnx(dataset=None, onnx_file_name="", num_class=3, assessment_method="accuracy", target_device="GPU"):
    """ do eval for onnx model"""
    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(num_class)
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")

    columns_list = ["input_ids", "input_mask", "token_type_id", "label_ids"]

    if not os.path.exists(onnx_file_name):
        raise ValueError("ONNX file not exists, please check onnx file has been saved and whether the "
                         "export_file_name is correct.")

    session, input_name_0, input_name_1, input_name_2, output_name_0 = create_session(onnx_file_name, target_device)


    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data

        x0 = input_ids.asnumpy()
        x1 = input_mask.asnumpy()
        x2 = token_type_id.asnumpy()

        result = session.run([output_name_0], {input_name_0: x0, input_name_1: x1, input_name_2: x2})
        logits = Tensor(result[0], dtype.float32)
        callback.update(logits, label_ids)

    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")


def run_classifier_onnx():
    """run classifier task for onnx model"""
    args_opt = parse_args()
    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do onnx evaluation task")
    assessment_method = args_opt.assessment_method.lower()
    ds = create_finetune_dataset(batch_size=args_opt.eval_batch_size,
                                 repeat_count=1,
                                 data_file_path=args_opt.eval_data_file_path,
                                 do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
    do_eval_onnx(ds, args_opt.onnx_file, args_opt.number_labels, assessment_method, args_opt.device_target)


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="run classifier")
    parser.add_argument("--task_type", type=str, default="chnsenticorp", choices=["chnsenticorp", "xnli", "dbqa"],
                        help="Task type, default is chnsenticorp")
    parser.add_argument("--assessment_method", type=str, default="accuracy", choices=["accuracy", "f1"],
                        help="Assessment method")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument('--onnx_file', type=str,
                        default="./ernie_finetune.onnx",
                        help='Onnx file path')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--number_labels", type=int, default=3, help="The number of class, default is 3.")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    args_opt = parser.parse_args()

    return args_opt


if __name__ == "__main__":
    run_classifier_onnx()
