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
"""trans data from multiple to one"""

import os
import math
from datetime import datetime
from multiprocessing import Process, Manager
import pickle
import numpy as np

def preprocess_raw_new(fidx, fea_utt_lst, label_utt_lst,
                       samples_dict_global, labels_dict_global, output_path):
    """merge single files into one

    :param samples_per_file: Number of samples per file
    :return: None
    """
    # initialize
    samples = []
    labels = []
    samples_dict = {}
    labels_dict = {}
    offset = 0
    offset_label = 0
    file_ind = fidx
    count = 0
    interval = 500
    for fea_path, label_path in zip(fea_utt_lst, label_utt_lst):
        fea = np.load(fea_path)
        label = np.load(label_path)
        nplabel = None
        if label.shape[0] != fea.shape[0]:
            print("shape not same：", label.shape[0], "!=", fea.shape[0])
            break
        else:
            nplabel = label.squeeze()
        fea_flat = fea.flatten()
        utt = fea_path[fea_path.rfind('/')+1:]
        samples_dict[utt[utt.rfind('/')+1:]] = (file_ind, offset, fea_flat.shape[0])
        labels_dict[utt[utt.rfind('/')+1:]] = (file_ind, offset_label, nplabel.shape[0])
        samples.append(fea_flat)
        labels.append(nplabel)
        offset += fea_flat.shape[0]
        offset_label += nplabel.shape[0]
        count += 1
        if count % interval == 0:
            print('process', fidx, count)

    labels = np.hstack(labels)
    np.save(os.path.join(output_path, f"{file_ind}_label.npy"), labels)
    samples = np.hstack(samples)
    np.save(os.path.join(output_path, f"{file_ind}.npy"), samples)
    print("save to", os.path.join(output_path))
    samples_dict_global.update(samples_dict)
    labels_dict_global.update(labels_dict)
    print('process', fidx, 'done')

def data_trans_dp(datasetPath, dataSavePath):
    fea_lst = os.path.join(datasetPath, "fea.lst")
    label_lst = os.path.join(datasetPath, "label.lst")
    print("fea_lst, label_lst:", fea_lst, label_lst)
    fea_utt_lst = []
    label_utt_lst = []
    with open(os.path.join(datasetPath, "fea.lst"), 'r') as fp:
        for line in fp:
            fea_utt_lst.append(os.path.join(datasetPath, line.strip()))
    with open(os.path.join(datasetPath, "label.lst"), 'r') as fp:
        for line in fp:
            label_utt_lst.append(os.path.join(datasetPath, line.strip()))

    print("total length of fea, label:", len(fea_utt_lst), len(label_utt_lst))

    fea_utt_lst_new = []
    label_utt_lst_new = []
    epoch_len = 73357
    for idx in range(len(fea_utt_lst)):
        if (idx+1)%epoch_len == 0:
            continue
        fea_utt_lst_new.append(fea_utt_lst[idx])
        label_utt_lst_new.append(label_utt_lst[idx])

    print(len(fea_utt_lst_new), len(label_utt_lst_new))
    fea_utt_lst = fea_utt_lst_new
    label_utt_lst = label_utt_lst_new

    samples_per_file = 4000
    total_process_num = math.ceil(len(fea_utt_lst) / samples_per_file)
    print('samples_per_file, total_process_num:',
          samples_per_file, total_process_num)
    samples_dict = Manager().dict()
    labels_dict = Manager().dict()

    thread_num = 10
    print(datetime.now().strftime("%m-%d-%H:%M:%S"))
    batchnum = math.ceil(total_process_num / thread_num)
    print('batch num:', batchnum)
    print("press Enter to continue...")
    input()
    for batchid in range(batchnum):
        threadlist = []
        for idx in range(thread_num):
            start = (batchid * thread_num + idx) * samples_per_file
            end = (batchid * thread_num + idx + 1) * samples_per_file
            if start >= len(fea_utt_lst):
                break
            if end > len(fea_utt_lst):
                end = len(fea_utt_lst)
            print(batchid * thread_num + idx, 'start, end:', start, end)
            p = Process(target=preprocess_raw_new,
                        args=(batchid * thread_num + idx, fea_utt_lst[start:end],
                              label_utt_lst[start:end], samples_dict, labels_dict, dataSavePath))
            p.start()
            threadlist.append(p)
        for p in threadlist:
            p.join()
    print(datetime.now().strftime("%m-%d-%H:%M:%S"))
    pickle.dump(dict(samples_dict), open(os.path.join(dataSavePath, f"ind_sample.p"), "wb"))
    pickle.dump(dict(labels_dict), open(os.path.join(dataSavePath, f"ind_label.p"), "wb"))
