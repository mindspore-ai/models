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
"""generate csv"""
import argparse
import os
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument("--data_url", type=str, default='')
parser.add_argument("--csv_dir", type=str, default='')
parser.add_argument("--output_dir", type=str, default='')
parser.add_argument("--triplet_num", type=int, default=10000)

args = parser.parse_args()


class TripletFaceDataset:

    def __init__(self, root_dir, csv_name, output_dir_, num_triplets):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.output_dir = output_dir_
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

        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        list_rows = []
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
            row = {'anc_id': face_classes[pos_class][ianc], 'pos_id': face_classes[pos_class][ipos],
                   'neg_id': face_classes[neg_class][ineg],
                   'pos_class': pos_class, 'neg_class': neg_class, 'pos_name': pos_name, 'neg_name': neg_name}
            list_rows.append(row)
        dataframe = pd.DataFrame(list_rows)
        dataframe.to_csv(path_or_buf=output_dir, index=False)

if __name__ == '__main__':
    output_dir = os.path.join(args.output_dir, "trp.csv")
    face_dataset = TripletFaceDataset(root_dir=args.data_url,
                                      csv_name=args.csv_namm,
                                      output_dir_=args.output_dir,
                                      num_triplets=args.triplet_num)
