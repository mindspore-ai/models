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
"""data process"""
import os
import pickle
import numpy as np

np.random.seed(58)

class DatasetGeneratorBatchEval:
    def __init__(self, data_path, read_limit=5000000):
        self.batchlist = []
        self.index_sample = {}
        self.memmaps_sample = {}
        self.memmaps_label = {}
        self.reads = 0
        self.read_limit = read_limit
        dataset_index = pickle.load(open(os.path.join(data_path, f"ind_sample.p"), "rb"))
        for utterance, (file_ind, offset, length) in dataset_index.items():
            file_path = os.path.join(data_path, f"{file_ind}.npy")
            self.index_sample[utterance] = (file_path, offset, length)
            self.batchlist.append(utterance)

    def __getitem__(self, index):
        utt = self.batchlist[index]
        utt_path, offset_utt, length_utt = self.index_sample[utt]
        if utt_path not in self.memmaps_sample:
            self.memmaps_sample[utt_path] = np.load(utt_path, mmap_mode="r")
        self.reads += 1
        if self.reads >= self.read_limit:
            self.flush_memmaps()
        fea = self.memmaps_sample[utt_path][offset_utt:offset_utt + length_utt]
        label = utt
        return fea.reshape((1, 301, 80)), label
    def flush_memmaps(self):
        for file_path in self.memmaps_sample:
            self.memmaps_sample[file_path] = None
            self.memmaps_sample[file_path] = np.load(file_path, mmap_mode="r")
        self.reads = 0
    def __len__(self):
        return len(self.batchlist)

class DatasetGenerator:
    def __init__(self, data_dir, drop=True):
        self.data = []
        self.label = []
        filelist = os.path.join(data_dir, "fea.lst")
        labellist = os.path.join(data_dir, "label.lst")
        with open(filelist, 'r') as fpF:
            for file in fpF:
                self.data.append(os.path.join(data_dir, file.strip()))
        with open(labellist, 'r') as fpL:
            for label in fpL:
                self.label.append(os.path.join(data_dir, label.strip()))
        if drop:
            self.data.pop()
            self.label.pop()
        print("dataset init ok, total len:", len(self.data))


    def __getitem__(self, index):
        npdata = np.load(self.data[index])
        nplabel = np.load(self.label[index]).tolist()
        return npdata, nplabel[0]

    def __len__(self):
        return len(self.data)

class DatasetGeneratorTrain:
    def __init__(self, data_dir, drop=False):
        self.data = []
        self.label = []
        filelist = os.path.join(data_dir, "fea.lst")
        labellist = os.path.join(data_dir, "label.lst")
        with open(filelist, 'r') as fpF:
            for file in fpF:
                self.data.append(data_dir + '/' + file.strip())
        with open(labellist, 'r') as fpL:
            for label in fpL:
                self.label.append(data_dir + '/' + label.strip())
        if drop:
            self.data.pop()
            self.label.pop()
        print("Load dataset ok, total len:", len(self.data))


    def __getitem__(self, index):
        npdata = np.load(self.data[index])
        nplabel = np.load(self.label[index])
        return npdata, np.squeeze(nplabel)

    def __len__(self):
        return len(self.data)

class DatasetGeneratorBatch:
    def __init__(self, data_paths, read_limit=5000000):
        self.batchlist = []
        self.index_sample = {}
        self.index_label = {}
        self.memmaps_sample = {}
        self.memmaps_label = {}
        self.reads = 0
        self.read_limit = read_limit
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        for data_path in data_paths:
            dataset_index = pickle.load(open(os.path.join(data_path, f"ind_sample.p"), "rb"))
            local_batchlist = []
            for utterance, (file_ind, offset, length) in dataset_index.items():
                file_path = os.path.join(data_path, f"{file_ind}.npy")
                self.index_sample[utterance] = (file_path, offset, length)
                local_batchlist.append(utterance)
            label_index = pickle.load(open(os.path.join(data_path, f"ind_label.p"), "rb"))
            for utterance, (file_ind, offset, length) in label_index.items():
                file_path = os.path.join(data_path, f"{file_ind}_label.npy")
                self.index_label[utterance] = (file_path, offset, length)
            local_batchlist.sort()
            self.batchlist += local_batchlist
    def __getitem__(self, index):
        utt = self.batchlist[index]
        utt_path, offset_utt, length_utt = self.index_sample[utt]
        label_path, offset_l, length_l = self.index_label[utt]
        if utt_path not in self.memmaps_sample:
            self.memmaps_sample[utt_path] = np.load(utt_path, mmap_mode="r")
        if label_path not in self.memmaps_label:
            self.memmaps_label[label_path] = np.load(label_path, mmap_mode="r")
        self.reads += 1
        if self.reads >= self.read_limit:
            self.flush_memmaps()
        fea = self.memmaps_sample[utt_path][offset_utt:offset_utt + length_utt]
        label = self.memmaps_label[label_path][offset_l:offset_l + length_l]
        return fea.reshape((192, 301, 80)), label
    def flush_memmaps(self):
        for file_path in self.memmaps_sample:
            self.memmaps_sample[file_path] = None
            self.memmaps_sample[file_path] = np.load(file_path, mmap_mode="r")
        for label_path in self.memmaps_label:
            self.memmaps_label[label_path] = None
            self.memmaps_label[label_path] = np.load(label_path, mmap_mode="r")
        self.reads = 0
    def __len__(self):
        return len(self.batchlist)
if  __name__ == "__main__":
    dataset_dir = ""
    dsg = DatasetGenerator(dataset_dir)
