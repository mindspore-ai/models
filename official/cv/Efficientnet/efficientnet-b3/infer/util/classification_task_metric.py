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
""" classification_task_metric.py """

import os
import sys
import json
import argparse
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def get_ground_truth_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = dict()
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            label_list = line.strip().split(" ")
            img_name = label_list[0].split(".")[0]
            img_lab = label_list[-1]
            img_gt_dict[img_name] = img_lab
    return img_gt_dict


def load_statistical_predict_result(filepath):
    """
    function:
    the prediction esult file data extraction
    input:
    result file:filepath
    output:
    n_label:numble of label
    data_vec: the probabilitie of prediction in the 1000
    :return: probabilities, numble of label, in_type, color
    """
    with open(filepath, 'r')as f:
        label_list = f.readline().strip().split(" ")
        n_label = len(label_list)
        data_vec = np.zeros((len(label_list)), dtype=np.float32)
        if n_label != 0:
            for ind, cls_ind in enumerate(label_list):
                data_vec[ind] = np.int(cls_ind)
    return data_vec, n_label


def create_visualization_statistical_result(prediction_file_path,
                                            result_store_path,
                                            img_gt_dict, topn=5):
    """ create_visualization_statistical_result """
    writer = open(result_store_path, 'w')
    table_dict = dict()
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []
    count = 0
    count_hit = np.zeros(topn)
    for tfile_name in os.listdir(prediction_file_path):
        count += 1
        infer_file_name = tfile_name.split('.')[0]
        index = infer_file_name.rfind('_')
        img_name = infer_file_name[:index]
        filepath = os.path.join(prediction_file_path, tfile_name)

        prediction, n_labels = load_statistical_predict_result(filepath)
        gt = img_gt_dict[img_name]
        if n_labels == 1001:
            real_label = int(gt) + 1
        else:
            real_label = int(gt)
        res_cnt = min(len(prediction), topn)
        for i in range(res_cnt):
            if str(real_label) == str(int(prediction[i])):
                count_hit[i] += 1
                break
    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        if count == 0:
            accuracy = 0
        else:
            accuracy = np.cumsum(count_hit) / count
        for i in range(res_cnt):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer, indent=4)
    writer.close()


def run():
    """ run """
    parser = argparse.ArgumentParser(description="input parameter.")
    parser.add_argument("--result_path", type=str,
                        help="the path of infer result", default="./result")
    parser.add_argument("--annotation_file", type=str,
                        help="annotation file of dataset.",
                        default="val_label.txt")
    parser.add_argument("--json_file", type=str,
                        help="output json file to store the results.",
                        default="./result.json")
    args = parser.parse_args()
    img_label_dict = get_ground_truth_fromtxt(args.annotation_file)
    create_visualization_statistical_result(args.result_path,
                                            args.json_file, img_label_dict,
                                            topn=5)


if __name__ == '__main__':
    run()
