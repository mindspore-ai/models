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
"""Data_loader_online"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset as de


class TripletFaceDataset:

    def __init__(self, root_dir, csv_name, num_triplets):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
        print("===init TripletFaceDataset===", flush=True)

    @staticmethod
    def generate_triplets(df, num_triplets):

        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        for _ in range(num_triplets):

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name])

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
                   train_csv_name, valid_csv_name,
                   num_train_triplets, num_valid_triplets,
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
        raise ValueError("mode should be in", data_loaders.keys())

    if mode == "train":
        face_dataset = TripletFaceDataset(root_dir=train_root_dir,
                                          csv_name=train_csv_name,
                                          num_triplets=num_train_triplets)
        sampler2 = de.RandomSampler(replacement=False, num_samples=10000)
        #sampler = de.DistributedSampler(group_size, rank, shuffle=shuffle)
        dataloaders[mode] = de.GeneratorDataset(face_dataset,
                                                dataset_column_names,
                                                num_samples=10000,
                                                num_parallel_workers=num_workers,
                                                python_multiprocessing=False)
        dataloaders[mode].add_sampler(sampler2)
        dataloaders[mode] = dataloaders[mode].map(input_columns=["anc_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["pos_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["neg_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].batch(batch_size, num_parallel_workers=1, drop_remainder=True)
        data_size1 = len(face_dataset)
    elif mode == "train_valid":
        face_dataset = TripletFaceDataset(root_dir=train_root_dir,
                                          csv_name=train_csv_name,
                                          num_triplets=num_train_triplets)
        sampler = None
        dataloaders[mode] = de.GeneratorDataset(face_dataset,
                                                dataset_column_names,
                                                num_samples=10000,
                                                num_parallel_workers=num_workers,
                                                python_multiprocessing=False)
        dataloaders[mode] = dataloaders[mode].map(input_columns=["anc_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["pos_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["neg_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].batch(batch_size, num_parallel_workers=32, drop_remainder=True)
        data_size1 = len(face_dataset)
    else:
        face_dataset = TripletFaceDataset(root_dir=valid_root_dir,
                                          csv_name=valid_csv_name,
                                          num_triplets=num_valid_triplets)
        sampler = None
        dataloaders[mode] = de.GeneratorDataset(face_dataset, column_names=dataset_column_names,
                                                sampler=sampler, num_parallel_workers=num_workers)
        dataloaders[mode] = dataloaders[mode].map(input_columns=["anc_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["pos_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].map(input_columns=["neg_img"], operations=data_transforms[mode])
        dataloaders[mode] = dataloaders[mode].batch(batch_size, num_parallel_workers=32, drop_remainder=True)
        data_size1 = len(face_dataset)

    return dataloaders, data_size1
