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
"""Utils"""

import os
import sys
import json
import logging
from datetime import datetime

import scipy.stats
import numpy as np


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)


def setup_default_logging(args, default_level=logging.INFO,
                          log_format="%(asctime)s - %(levelname)s -  %(message)s"):

    logger = logging.getLogger('')

    if not os.path.exists(args.exp_dir) and args.exp_dir != '':
        os.makedirs(args.exp_dir)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(args.exp_dir, f'{time_str()}.log'),
        format=log_format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    return "Local Job"


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(folder, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    folder = os.path.expanduser(folder)
    if not (extensions is None) ^ (is_valid_file is None):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def check_file(x):
            return has_file_allowed_extension(x, extensions)
    else:
        check_file = is_valid_file
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(folder, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if check_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def find_classes(folder):
    """
    Finds the class folders in a dataset.

    Args:
        folder (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (folder), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(folder) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def getEntropy(logist: list = None) -> float:
    """according the logist get the entropy"""

    info_entro = 0.0
    for each_probe in logist:
        info_entro = info_entro - each_probe * np.log2(each_probe + 1e-8)
    return info_entro


def getDiversity(logistA: list, logistB: list) -> float:
    """cacualte the  KL between the logistA and logistB"""

    info_kl = scipy.stats.entropy(logistA, logistB)
    return info_kl


def getDiversity_final(logistA: list, logistB: list) -> float:
    """cacualte the  KL between the logistA and logistB"""
    info_kl_ = []
    for a_prob, b_probe in zip(logistA, logistB):
        same = a_prob * np.log2(a_prob + 1e-8) + b_probe * np.log2(b_probe + 1e-8)
        diff = a_prob * np.log2(b_probe + 1e-8) + b_probe * np.log2(a_prob + 1e-8)
        info_kl_.append(same - diff)
    result_all = np.sum(info_kl_)
    return result_all


def getMatrix(features: list = None, preserve: float = 0.25) -> int:
    """according the features get the matrix used for selecting the sample hard and diversity"""

    logits = softmax(features)

    # get the max index and logits (the minority subordinate to the majority)
    list_index = []
    for each in range(len(logits)):
        index_cur = logits[each].argmax()
        list_index.append(index_cur)

    list_set = set(list_index)
    max_count = 0
    index_need = 0
    for item in list_set:
        item_count = list_index.count(item)
        if item_count > max_count:
            max_count = item_count
            index_need = item

    valid_count = int(len(logits) * preserve)
    matrix = np.zeros(shape=(valid_count, valid_count))

    mode_most = []
    if max_count >= valid_count:
        for each in range(len(list_index)):
            if list_index[each] == index_need:
                mode_most.append(logits[each])

    for i in range(valid_count):
        for j in range(valid_count):
            matrix[i][j] = getEntropy(logits[i]) if i == j else getDiversity(logits[i], logits[j])

    return np.sum(matrix.__abs__())


def getMatrix_final(features: list = None, preserve: float = 0.25) -> int:
    """according the features get the matrix used for selecting the sample hard and diversity"""

    logits = softmax(features)
    index_max = np.argsort(logits)[:, -1]
    value_max = []
    for each_index in range(len(logits)):
        cur_max_index = index_max[each_index]
        cur_value = logits[each_index][cur_max_index]
        value_max.append(cur_value)

    value_max_sorted_desc = np.argsort(value_max)[::-1]
    sorted_logist = [logits[indx] for indx in value_max_sorted_desc]

    list_set = set(index_max)
    max_count = 0
    for item in list_set:
        item_count = index_max.tolist().count(item)
        if item_count > max_count:
            max_count = item_count

    valid_count = int(len(logits) * preserve)
    matrix = np.zeros(shape=(valid_count, valid_count))
    prob_max = max_count / len(logits)

    for i in range(valid_count):
        for j in range(valid_count):
            # use the top a percent of the patches when prob_max > 0.5 else use the bottom a percent of the patches
            logist_i = sorted_logist[i] if prob_max > 0.5 else sorted_logist[len(sorted_logist) - i - 1]
            logist_j = sorted_logist[j] if prob_max > 0.5 else sorted_logist[len(sorted_logist) - j - 1]
            matrix[i][j] = getEntropy(logist_i) if i == j else getDiversity_final(logist_i, logist_j)
    return np.sum(matrix)


def softmax(x):
    """
    softmax函数实现
    参数：
    x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度
    返回：
    softmax计算结果
    """
    row_max = x.max(axis=1).reshape(-1, 1)
    x -= row_max
    x_exp = np.exp(x)
    s = x_exp / np.sum(x_exp, axis=1, keepdims=True)

    return s


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def txt_to_dict(txt_path):
    return_dict = {}
    txt_file = open(txt_path, 'r')

    line = txt_file.readline()
    while line:
        name_value = line.rstrip().split(" ")
        name = name_value[0]
        value = float(name_value[1])
        return_dict[name] = value

        line = txt_file.readline()
    return return_dict


def dict_save_json(dict_self, json_path):
    with open(json_path, 'w') as f:
        json.dump(dict_self, f)


def refine_json_again(class_to_id, base_json_path, activate_json_path,
                      targe_save_path, rate=0.15, unlable_contain_all=False):
    if unlable_contain_all:
        data_base = read_json(base_json_path)
        labeled_samples = data_base['labeled_samples']
        unlabeled_samples = data_base['unlabeled_samples']
        need_num = int(len(unlabeled_samples) * rate)
        data_activate = read_json(activate_json_path)
        class_to_idx = read_json(class_to_id)['class_to_idx']
        i = 0
        for each in data_activate:
            each_name = each[0]
            class_name = os.path.basename(each[0]).split("_")[0]
            each_item = [each_name, class_to_idx[class_name]]
            if each_item in labeled_samples:
                continue
            else:
                labeled_samples.append((each_name, class_to_idx[class_name]))
                i = i + 1
            if i >= need_num:
                break
    else:
        data_base = read_json(base_json_path)
        labeled_samples = data_base['labeled_samples']
        unlabeled_samples = data_base['labeled_samples'] + data_base['unlabeled_samples']

        need_num = int(len(unlabeled_samples) * rate)

        data_activate = read_json(activate_json_path)
        class_to_idx = read_json(class_to_id)['class_to_idx']

        for each in data_activate[:need_num]:
            each_name = each[0]
            class_name = os.path.basename(each[0]).split("_")[0]
            labeled_samples.append((each_name, class_to_idx[class_name]))

    annotation = {'labeled_samples': labeled_samples, 'unlabeled_samples': unlabeled_samples}
    dict_save_json(annotation, targe_save_path)
    logging.info("done")


def replace_dict(json_base, json_new, str_ori, str_replace):
    data_base = read_json(json_base)
    labeled_samples = data_base['labeled_samples']
    unlabeled_samples = data_base['unlabeled_samples']

    labeled_samples_new = []
    unlabeled_samples_new = []

    for each in labeled_samples:
        each_new = each[0].replace(str_ori, str_replace)
        labeled_samples_new.append((each_new, each[1]))
        if not os.path.exists(each_new):
            logging.info(each_new)

    for each in unlabeled_samples:
        each_new = each[0].replace(str_ori, str_replace)
        unlabeled_samples_new.append((each_new, each[1]))
        if not os.path.exists(each_new):
            logging.info(each_new)

    annotation = {'labeled_samples': labeled_samples_new, 'unlabeled_samples': unlabeled_samples_new}
    dict_save_json(annotation, json_new)
    logging.info("done")
