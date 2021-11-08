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
"""preprocess"""
import os
import argparse
import shutil
import numpy as np

from src.model_utils.config import config as cfg
from src.dataset import create_dataset
from src.dataset import convert_to_mindrecord

parser = argparse.ArgumentParser(description='textrcnn')
parser.add_argument('--task', type=str, \
    help='the data preprocess task, including dataset_split.', default='dataset_split')
parser.add_argument('--data_dir', type=str, help='the source dataset directory.', default='./data_src')

args = parser.parse_args()


def convert_encoding(path_name, file_name):
    """convert encoding method"""
    f = open(file_name, 'r', encoding='iso-8859-1')
    tmp_name = os.path.join(path_name, "tmp")
    f_tmp = open(tmp_name, 'w+', encoding='utf-8')
    for line in f:
        f_tmp.write(line)
    for line in f_tmp:
        print(line)
    f.close()
    f_tmp.close()
    os.remove(file_name)
    os.rename(tmp_name, file_name)


def dataset_split(label):
    """dataset_split api"""
    # label can be 'pos' or 'neg'
    pos_samples = []
    pos_path = os.path.join(args.data_dir, "rt-polaritydata")
    pos_file = os.path.join(pos_path, "rt-polarity." + label)

    convert_encoding(pos_path, pos_file)

    pfhand = open(pos_file, encoding='utf-8')
    pos_samples += pfhand.readlines()
    pfhand.close()
    np.random.seed(0)
    perm = np.random.permutation(len(pos_samples))
    perm_train = perm[0:int(len(pos_samples) * 0.9)]
    perm_test = perm[int(len(pos_samples) * 0.9):]
    pos_samples_train = []
    pos_samples_test = []
    for pt in perm_train:
        pos_samples_train.append(pos_samples[pt])
    for pt in perm_test:
        pos_samples_test.append(pos_samples[pt])

    if not os.path.exists(os.path.join(cfg.data_root, 'train')):
        os.makedirs(os.path.join(cfg.data_root, 'train'))
    if not os.path.exists(os.path.join(cfg.data_root, 'test')):
        os.makedirs(os.path.join(cfg.data_root, 'test'))

    f = open(os.path.join(cfg.data_root, 'train', label), "w")
    f.write(''.join(pos_samples_train))
    f.close()

    f = open(os.path.join(cfg.data_root, 'test', label), "w")
    f.write(''.join(pos_samples_test))
    f.close()


def get_bin():
    """generate bin files."""
    ds_eval = create_dataset(cfg.preprocess_path, cfg.batch_size, False)
    img_path = os.path.join(cfg.pre_result_path, "00_feature")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    label_list = []

    for i, data in enumerate(ds_eval.create_dict_iterator(output_numpy=True)):
        file_name = "textrcnn_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(img_path, file_name)

        data["feature"].tofile(file_path)
        label_list.append(data["label"])

    np.save(os.path.join(cfg.pre_result_path, "label_ids.npy"), label_list)
    print("=" * 20, "bin files finished", "=" * 20)


if __name__ == '__main__':
    # split
    if args.task == "dataset_split":
        dataset_split('pos')
        dataset_split('neg')

    # convert to mindrecord
    print("============== Starting Data Pre-processing ==============")
    if os.path.exists(cfg.preprocess_path):
        shutil.rmtree(cfg.preprocess_path)
    os.mkdir(cfg.preprocess_path)
    convert_to_mindrecord(cfg.embed_size, cfg.data_root, cfg.preprocess_path, cfg.emb_path)

    # get bin
    get_bin()
