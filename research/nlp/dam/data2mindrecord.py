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
"""Data format is converted to MindRecord"""
import os
import ast
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from mindspore import context
from mindspore import dataset as ds
from mindspore.mindrecord import FileWriter


def unison_shuffle(r_data, seed=None):
    """
    Shuffle data
    """
    if seed is not None:
        np.random.seed(seed)

    y = np.array(r_data[b'y'])
    c = np.array(r_data[b'c'])
    r = np.array(r_data[b'r'])

    assert len(y) == len(c) == len(r)
    p = np.random.permutation(len(y))
    print(p)
    shuffle_data = {b'y': y[p], b'c': c[p], b'r': r[p]}
    return shuffle_data


def split_c(c, split_id):
    """
    Split
    c is a list, example context
    split_id is a integer, conf[_EOS_]
    return nested list
    """
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns


def normalize_length(_list, length, cut_type='tail'):
    """_
    list is a list or nested list, example turns/r/single turn c
    cut_type is head or tail, if _list len > length is used
    return a list len=length and min(read_length, length)
    """
    real_length = len(_list)
    out_list = _list
    out_length = real_length
    if real_length == 0:
        out_list = [0] * length
        out_length = 0
    elif real_length <= length:
        if not isinstance(_list[0], list):
            _list.extend([0] * (length - real_length))
        else:
            _list.extend([[]] * (length - real_length))
        out_list = _list
        out_length = real_length
    else:
        if cut_type == 'head':
            out_list = _list[:length]
            out_length = length
        if cut_type == 'tail':
            out_list = _list[-length:]
            out_length = length
    return out_list, out_length


def produce_one_sample(_data, index, split_id, max_turn_num, max_turn_len, turn_cut_type='tail',
                       term_cut_type='tail'):
    '''
    max_turn_num=10
    max_turn_len=50
    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
    '''
    c = _data[b'c'][index]
    r = _data[b'r'][index][:]
    y = _data[b'y'][index]

    turns = split_c(c, split_id)
    # normalize turns_c length, nor_turns length is max_turn_num
    nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)

    nor_turns_nor_c = []
    term_len = []
    # nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns:
        # nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len


def data2mindrecord(orig_data, target_data, mode_, config_):
    """
    Convert Dataset To Mindrecord

    :param orig_data: Path of raw data.
    :param target_data: Path to destination data
    :param mode_: Train, Val, Test
    :param config_: Parameters for processing data
    :return:
    """
    print('config.EOS: ', config_.EOS)
    MINDRECORD_FILE = target_data
    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        if os.path.exists(MINDRECORD_FILE + '.db'):
            os.remove(MINDRECORD_FILE + '.db')

    writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
    schema = {"turns": {"type": "int32", "shape": [config_.max_turn_num, config_.max_turn_len]},
              "turn_len": {"type": "int32", "shape": [-1]},
              "response": {"type": "int32", "shape": [-1]},
              "response_len": {"type": "int32", "shape": [-1]},
              "label": {"type": "int32", "shape": [-1]}}
    writer.add_schema(schema, mode_ + "dataset")

    with open(orig_data, "rb") as f:
        print("Loading .pkl file.")
        train, val, test = pickle.load(f, encoding="bytes")
        print('train_data.len: ', len(train[b'y']))
        print('eval_data.len: ', len(val[b'y']))
        print('test_data.len: ', len(test[b'y']))
    if mode_ == "train":
        data = train
    elif mode_ == "val":
        data = val
    else:
        data = test

    if config.shuffle:
        print("Using shuffle.")
        data = unison_shuffle(r_data=data, seed=config.seed)

    max_turn_num = config_.max_turn_num
    max_turn_len = config_.max_turn_len
    EOS = config_.EOS
    print('EOS: ', EOS)

    data_len = int(len(data[b'y']))
    print('data_len: ', data_len)
    data_list = []
    count = 0
    for index in tqdm(range(data_len)):
        count += 1
        y, nor_turns_nor_c, nor_r, _, term_len, r_len = produce_one_sample(data, index, EOS, max_turn_num, max_turn_len,
                                                                           turn_cut_type='tail', term_cut_type='tail')
        sample = {"turns": np.array(nor_turns_nor_c),
                  "turn_len": np.array(term_len),
                  "response": np.array(nor_r),
                  "response_len": np.array(r_len),
                  "label": np.array(y)}
        data_list.append(sample)
        if count % 100 == 0:
            writer.write_raw_data(data_list)
            data_list.clear()
        if count % 100000 == 0:
            print('Have handle {}w lines.'.format((count / 100000) * 10))
    if data_list:
        writer.write_raw_data(data_list)
    print('total {} lines.'.format(count))
    writer.commit()
    print("read over")


def precess_data_args():
    """
    Precessing Data Args.
    """
    parser = argparse.ArgumentParser("DAM Training Args")
    parser.add_argument('--data_name', type=str, default="ubuntu", help='The data name.')  # douban: douban
    parser.add_argument('--device_target', type=str, default="Ascend", help="run platform, only support Ascend")
    parser.add_argument('--device_id', type=int, default=0)
    # net args
    parser.add_argument('--max_turn_num', type=int, default=9)
    parser.add_argument('--max_turn_len', type=int, default=50)
    parser.add_argument('--EOS', type=int, default=28270)  # 1 for douban data
    # path
    parser.add_argument('--data_root', type=str, default="./data/ubuntu/")  # douban: ./data/douban
    parser.add_argument('--raw_data', type=str, default="data.pkl")
    parser.add_argument('--mode', required=True, type=str, default="train")
    parser.add_argument('--shuffle', type=ast.literal_eval, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_data', type=ast.literal_eval, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = precess_data_args()
    if config.data_name == "ubuntu":
        config.EOS = 28270
    elif config.data_name == "douban":
        config.EOS = 1
    else:
        raise RuntimeError('{} does not exist'.format(config.data_name))
    print("args: ", config, '\n')

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=config.device_id)

    data_file = os.path.join(config.data_root, config.raw_data)
    print("raw data: ", data_file)

    mode = config.mode
    target_file = os.path.join(config.data_root, ("data_" + mode + ".mindrecord"))
    print('mode: ', mode)
    print('target data: ', target_file)

    print("Starting processing the data.")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    data2mindrecord(orig_data=data_file, target_data=target_file, mode_=mode, config_=config)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Succeed")

    if config.print_data:
        dataset = ds.MindDataset(target_file, columns_list=["turns", "turn_len", "response", "response_len", "label"],
                                 shuffle=False)
        dataset = dataset.batch(200)
        data_loader = dataset.create_dict_iterator()
        for i, d in enumerate(data_loader, start=1):
            print(i, d["turns"])
