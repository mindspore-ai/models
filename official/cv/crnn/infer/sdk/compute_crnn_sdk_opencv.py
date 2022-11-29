# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
import numpy as np

def read_label_and_pred(npu_data_path):
    """
    read label and predict
    :param npu_data_path:
    :return:
    """
    label_list = []
    npu_data_list = []
    all_file = os.listdir(npu_data_path)
    all_file.sort()
    for result_file in all_file:
        if result_file.endswith(".json"):
            label = result_file.split("_")[-1].split(".")[0].lower()
            label_list.append(label)
            result_path = npu_data_path + "/" + result_file
            with open(result_path, "r") as f:
                load_dict = json.load(f)
                npu_str = load_dict["MxpiTextsInfo"][0]["text"][0]
                npu_data_list.append(npu_str)
    return label_list, npu_data_list


def compute_per_char(ground_truth, predictions):
    """
    compute per char accuracy
    :param ground_truth:
    :param predictions:
    :return:
    """
    accuracy = []
    for index, label in enumerate(ground_truth):
        prediction = predictions[index]
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp == prediction[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if prediction is None:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
    avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    print('PerChar Precision is {:5f}'.format(avg_accuracy))
    return avg_accuracy


def compute_full_sequence(ground_truth, predictions):
    """
    compute full sequence accuracy
    :param ground_truth:
    :param predictions:
    :return:
    """
    try:
        correct_count = 0
        mistake_count = [0] * 7
        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            if prediction == label:
                correct_count += 1
            else:
                mistake_count[int(index/100)] += 1
        avg_accuracy = correct_count / len(ground_truth)
        print("correct num: " + str(correct_count))
        print("total count: " + str(len(ground_truth)))
        print("mistake count: " + str(mistake_count))
    except ZeroDivisionError:
        if not predictions:
            avg_accuracy = 1
        else:
            avg_accuracy = 0
    print('infer accuracy is {:5f}'.format(avg_accuracy))
    return avg_accuracy


def compute_accuracy(ground_truth, predictions, mode='per_char'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param mode:
    :return: avg_label_accuracy
    """
    if mode == "per_char":
        avg_accuracy = compute_per_char(ground_truth, predictions)
    elif mode == 'full_sequence':
        avg_accuracy = compute_full_sequence(ground_truth, predictions)
    else:
        raise NotImplementedError(
            'Other accuracy compute model has not been implemented')

    return avg_accuracy


def main(args):
    """
    main function
    :param args:
    :return:
    """
    gt_data_list, npu_data_list = read_label_and_pred(args.npu_data_path)
    compute_accuracy(gt_data_list, npu_data_list, mode="per_char")
    compute_accuracy(gt_data_list, npu_data_list, mode="full_sequence")


def parse_args():
    """
    parse args
    :return:
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--npu_data_path', type=str,
                       default='./output/')
    return parse.parse_args()


if __name__ == '__main__':
    main(parse_args())
