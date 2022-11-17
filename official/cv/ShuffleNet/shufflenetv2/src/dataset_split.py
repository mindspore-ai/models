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
import os
import random
from shutil import rmtree, copy2

random.seed(1)
input_path = './data/flower_photos'
train_path = './data/flower_photos_split/train'
eval_path = './data/flower_photos_split/eval'


def dataset_split():
    if os.path.exists(train_path):
        rmtree(train_path)
    if os.path.exists(eval_path):
        rmtree(eval_path)

    os.makedirs(train_path)
    os.makedirs(eval_path)

    flowers = os.listdir(input_path)

    for flower in flowers:
        if os.path.isdir(os.path.join(input_path, flower)):
            flower_path = os.path.join(input_path, flower)
            images = os.listdir(flower_path)
            images_path = list()
            for image in images:
                image_path = os.path.join(flower_path, image)
                images_path.append(image_path)
            random.shuffle(images_path)
            offset = int(len(images_path) * 0.8)
            train_set = images_path[:offset]
            eval_set = images_path[offset:]
            aim_train_path = os.path.join(train_path, flower)
            aim_eval_path = os.path.join(eval_path, flower)
            os.makedirs(aim_train_path)
            os.makedirs(aim_eval_path)
            for image in train_set:
                print("copying " + image + " to " + aim_train_path)
                copy2(image, aim_train_path)
            for image in eval_set:
                print("copying " + image + " to " + aim_eval_path)
                copy2(image, aim_eval_path)


if __name__ == '__main__':
    dataset_split()
