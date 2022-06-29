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
"""Data_loader"""

import os
import csv
import numpy as np
from PIL import Image
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset as de


class TripletFaceDataset:

    def __init__(self, root_dir, triplets_path):
        self.root_dir = root_dir
        self.triplets_path = triplets_path
        self.training_triplets = self.get_triplets(triplets_path)
        print("===init TripletFaceDataset===", flush=True)

    def get_triplets(self, triplets_csv):
        triplets = list()
        with open(triplets_csv, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                triplets.append(row)
        return triplets

    def __getitem__(self, idx):
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_name), str(anc_id) + '.png')
        pos_img = os.path.join(self.root_dir, str(pos_name), str(pos_id) + '.png')
        neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id) + '.png')

        anc_img = Image.open(anc_img).convert("RGB")
        pos_img = Image.open(pos_img).convert("RGB")
        neg_img = Image.open(neg_img).convert("RGB")

        pos_class = np.array([pos_class]).astype(np.int32)
        neg_class = np.array([neg_class]).astype(np.int32)

        return (anc_img, pos_img, neg_img, pos_class, neg_class)

    def __len__(self):
        return len(self.training_triplets)


def get_dataloader(train_root_dir, valid_root_dir,
                   train_triplets, valid_triplets,
                   batch_size, num_workers, group_size,
                   rank, shuffle, mode='train'):

    data_transforms = {
        'train': [
            C.RandomResize(size=(224, 224)),
            C.RandomHorizontalFlip(),
            P.ToTensor(),
            P.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
        'train_valid': [
            C.RandomResize(size=(224, 224)),
            C.RandomHorizontalFlip(),
            P.ToTensor(),
            P.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
        'valid': [
            C.RandomResize(size=(224, 224)),
            P.ToTensor(),
            P.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]}


    dataset_column_names = ["anc_img", "pos_img", "neg_img", "pos_class", "neg_class"]

    dataloaders = {"train": None, "valid": None, "train_valid": None}
    if mode not in dataloaders:
        raise ValueError("mode should be in", dataloaders.keys())

    if mode == "train":
        face_dataset = TripletFaceDataset(root_dir=train_root_dir,
                                          triplets_path=train_triplets)
        print(train_root_dir)
        print(train_triplets)
        sampler1 = de.DistributedSampler(group_size, rank, shuffle=shuffle)
        sampler2 = de.RandomSampler(replacement=False, num_samples=10000)
        dataloaders[mode] = de.GeneratorDataset(face_dataset,
                                                dataset_column_names,
                                                #num_samples=10000,
                                                sampler=sampler2,
                                                num_parallel_workers=num_workers,
                                                python_multiprocessing=False)
        dataloaders[mode].add_sampler(sampler1)
        #dataloaders[mode].add_sampler(sampler2)
        dataloaders[mode] = dataloaders[mode].map(input_columns=["anc_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["pos_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["neg_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].batch(batch_size, num_parallel_workers=num_workers, drop_remainder=True)
        data_size1 = len(face_dataset)
        print("Dataset length:", data_size1, flush=True)
    elif mode == "train_valid":
        face_dataset = TripletFaceDataset(root_dir=train_root_dir,
                                          triplets_path=train_triplets)
        sampler1 = de.DistributedSampler(group_size, rank, shuffle=shuffle)
        sampler2 = de.RandomSampler(replacement=False, num_samples=10000)
        dataloaders[mode] = de.GeneratorDataset(face_dataset,
                                                dataset_column_names,
                                                num_samples=10000,
                                                num_parallel_workers=num_workers,
                                                python_multiprocessing=False)
        dataloaders[mode].add_sampler(sampler2)
        #dataloaders[mode].add_sampler(sampler1)
        dataloaders[mode] = dataloaders[mode].map(input_columns=["anc_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["pos_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["neg_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].batch(batch_size, num_parallel_workers=num_workers, drop_remainder=True)
        data_size1 = len(face_dataset)
    else:
        face_dataset = TripletFaceDataset(root_dir=valid_root_dir,
                                          triplets_path=valid_triplets)
        sampler = None
        dataloaders[mode] = de.GeneratorDataset(face_dataset, column_names=dataset_column_names,
                                                sampler=sampler, num_parallel_workers=num_workers)
        dataloaders[mode] = dataloaders[mode].map(input_columns=["anc_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["pos_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["neg_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].batch(batch_size, num_parallel_workers=num_workers, drop_remainder=True)
        data_size1 = len(face_dataset)

    return dataloaders, data_size1
