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
"""Split train and valid DeepID Dataset"""
import os
import os.path
import random

def fatch_pics_for_one_user(src_path, people_name):
    "fatch pics for one user"
    people_path = os.path.join(src_path, people_name)
    people_imgs = []
    for video_folder in os.listdir(people_path):
        for video_file_name in os.listdir(os.path.join(people_path, video_folder)):
            people_imgs.append(os.path.join(people_name, video_folder, video_file_name))
    random.shuffle(people_imgs)
    return people_imgs

def build_dataset(src_folder):
    "Build Dataset"
    total_people, total_picture = 0, 0
    test_people, valid_set, train_set = [], [], []
    label = 0

    for people_folder in os.listdir(src_folder):
        print(people_folder)
        people_imgs = fatch_pics_for_one_user(src_folder, people_folder)
        #people_imgs = fatch_pics_for_one_user(os.path.join(src_folder, people_folder))
        total_people += 1
        total_picture += len(people_imgs)
        if len(people_imgs) < 100:
            test_people.append(people_imgs)
        else:
            valid_set += zip(people_imgs[:10], [label]*10)
            train_set += zip(people_imgs[10:100], [label]*90)
            label += 1

    test_set = []
    for i, people_imgs in enumerate(test_people):
        for _ in range(5):
            same_pair = random.sample(people_imgs, 2)
            test_set.append((same_pair[0], same_pair[1], 1))
        for _ in range(5):
            j = i
            while j == i:
                j = random.randint(0, len(test_people)-1)
            test_set.append((random.choice(test_people[i]), random.choice(test_people[j]), 0))

    random.shuffle(test_set)
    random.shuffle(valid_set)
    random.shuffle(train_set)

    print('\tpeople\tpicture')
    print('total:\t%6d\t%7d' % (total_people, total_picture))
    print('test:\t%6d\t%7d' % (len(test_people), len(test_set)))
    print('valid:\t%6d\t%7d' % (label, len(valid_set)))
    print('train:\t%6d\t%7d' % (label, len(train_set)))
    return test_set, valid_set, train_set

def set_to_csv_file(data_set, file_name):
    "set to csv file"
    with open(file_name, "w") as f:
        for item in data_set:
            print(" ".join(map(str, item)), file=f)

if __name__ == '__main__':
    random.seed(7)
    src = "../data/crop_images_DB"
    test = "../data/test_set.csv"
    valid = "../data/valid_set.csv"
    train = "../data/train_set.csv"
    if not src.endswith('/'):
        src += '/'

    test_s, valid_s, train_s = build_dataset(src)
    set_to_csv_file(test_s, test)
    set_to_csv_file(valid_s, valid)
    set_to_csv_file(train_s, train)
