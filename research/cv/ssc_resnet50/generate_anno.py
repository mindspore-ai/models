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
"""Generate annotation json"""
import os
import argparse
import random
import json
import numpy as np

from src.utils import find_classes, make_dataset, IMG_EXTENSIONS

parser = argparse.ArgumentParser(description='generate annotation')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num_class', type=int, default=1000, help='percentage of labeled samples')

# dataset settings
parser.add_argument('--percent', type=int, default=10, choices=[10, 20, 25, 30, 40],
                    help='percentage of labeled samples')
parser.add_argument('--class_uniform', default=True, help='sample data uniform')
parser.add_argument('--annotation', type=str, help='annotation file')

args = parser.parse_args()


if __name__ == '__main__':
    classes, class_to_idx = find_classes(args.data)
    samples = make_dataset(args.data, class_to_idx, IMG_EXTENSIONS, None)
    classes_dirs = [os.path.join(args.data, path) for path in os.listdir(args.data)]
    classes_dirs = [path for path in classes_dirs if os.path.isdir(path)]
    all_classes = 0
    for classes_dir in classes_dirs:
        all_classes += len(os.listdir(classes_dir))
    label_per_class = args.percent * all_classes / 100 // args.num_class
    print("label_per_class size ", label_per_class)
    labeled_samples = []
    unlabeled_samples = []
    random.shuffle(samples)

    if args.class_uniform:
        print("uniform select the label sample for train")
        num_img = np.zeros(args.num_class)
        for i, (img, label) in enumerate(samples):
            if num_img[label] < label_per_class:
                labeled_samples.append((img, label))
                num_img[label] += 1
            else:
                unlabeled_samples.append((img, label))
    else:
        print("random select the label sample for train")
        for i, (img, label) in enumerate(samples):
            if i < int(label_per_class * args.num_class):
                labeled_samples.append((img, label))
            else:
                unlabeled_samples.append((img, label))

    annotation = {'labeled_samples': labeled_samples, 'unlabeled_samples': unlabeled_samples}
    with open(args.annotation, 'w') as f:
        json.dump(annotation, f)
    print("save the annotation to {0}".format(args.annotation))
