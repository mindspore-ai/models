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

import os
import time
import argparse
import glob
import pandas as pd



parser = argparse.ArgumentParser(description="Generating csv file for triplet loss!")
parser.add_argument('--dataroot', type=str, required=True,)
parser.add_argument('--csv_name', type=str, default="VGGFACE2.csv")
args = parser.parse_args()



def generate_csv_file(dataroot, csv_name):
    """Generates a csv file containing the image paths of the glint360k dataset for use in triplet selection in
    triplet loss training.

    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
    """
    print("\nLoading image paths ...")
    files = glob.glob(dataroot + "/*/*")

    start_time = time.time()
    list_rows = []

    print("Number of files: {}".format(len(files)))
    print("\nGenerating csv file ...")

    progress_bar = enumerate((files))
    # "datasets/people0/123.jpg"
    for _, file in progress_bar:

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))

        # Better alternative than dataframe.append()
        row = {'id': face_id, 'name': face_label}   # dict
        list_rows.append(row)


    dataframe = pd.DataFrame(list_rows)
    dataframe = dataframe.sort_values(by=['name', 'id']).reset_index(drop=True)


    # Encode names as categorical classes
    dataframe['class'] = pd.factorize(dataframe['name'])[0]
    dataframe.to_csv(path_or_buf=csv_name, index=False)


    elapsed_time = time.time()-start_time
    print("\nDone! Elapsed time: {:.2f} minutes.".format(elapsed_time/60))


if __name__ == '__main__':
    generate_csv_file(dataroot=args.dataroot, csv_name=args.csv_name)
