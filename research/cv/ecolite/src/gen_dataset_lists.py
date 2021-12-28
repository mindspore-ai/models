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

"""processing the raw data of the video datasets"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str,
                    choices=['something', 'jester', 'ucf101', 'hmdb51', 'activitynet_1.2', 'activitynet_1.3'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('--labels_path', type=str, default='data/dataset_labels/',
                    help="root directory holding the 20bn csv files: labels, train & validation")
parser.add_argument('--out_list_path', type=str, default='data/')
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y')
parser.add_argument('--num_split', type=int, default=1)
parser.add_argument('--shuffle', action='store_true', default=True)

args = parser.parse_args()

dataset = args.dataset
labels_path = args.labels_path
frame_path = args.frame_path

if dataset == 'something':
    dataset_name = 'something-something-v1'

    print('\nProcessing dataset: {}\n'.format(dataset))

    print('- Generating {}_category.txt ......'.format(dataset))
    with open(os.path.join(labels_path, '{}-labels.csv'.format(dataset_name))) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    open(os.path.join(args.out_list_path, '{}_category.txt'.format(dataset)), 'w').write('\n'.join(categories))
    print('- Saved as:', os.path.join(args.out_list_path, '{}_category.txt!\n'.format(dataset)))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['{}-validation.csv'.format(dataset_name), '{}-train.csv'.format(dataset_name)]
    files_output = ['{}_val.txt'.format(dataset), '{}_train.txt'.format(dataset)]
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(os.path.join(labels_path, filename_input)) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(os.path.join(str(dict_categories[items[1]])))
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(frame_path, curFolder))
            output.append('{} {} {}'.format(os.path.join(frame_path, curFolder), len(dir_files), curIDX))
            if i % 1000 == 0:
                print('- Generating {} ({}/{})'.format(filename_output, i, len(folders)))
        with open(os.path.join(args.out_list_path, filename_output), 'w') as f:
            f.write('\n'.join(output))
        print('- Saved as:', os.path.join(args.out_list_path, '{}!\n'.format(filename_output)))

elif dataset == 'ucf101':
    from pyActionRecog import parse_directory, build_split_list
    from pyActionRecog import parse_split_file

    rgb_p = args.rgb_prefix
    flow_x_p = args.flow_x_prefix
    flow_y_p = args.flow_y_prefix
    num_split = args.num_split
    out_path = args.out_list_path
    shuffle = args.shuffle

    # operation
    print('\nProcessing dataset {}:\n'.format(dataset))
    split_tp = parse_split_file(dataset)
    f_info = parse_directory(frame_path, rgb_p, flow_x_p, flow_y_p)

    print('- Writing list files for training/testing')
    for i in range(max(num_split, len(split_tp))):
        lists = build_split_list(split_tp, f_info, i, shuffle)
        open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][0])
        open(os.path.join(out_path, '{}_rgb_val_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][1])

    print('- List files successfully saved to "data/" folder!\n')
elif dataset == 'hmdb51':

    from pyActionRecog import parse_directory, build_split_list
    from pyActionRecog import parse_split_file

    rgb_p = args.rgb_prefix
    flow_x_p = args.flow_x_prefix
    flow_y_p = args.flow_y_prefix
    num_split = args.num_split
    out_path = args.out_list_path
    shuffle = args.shuffle

    # operation
    print('\nProcessing dataset {}:\n'.format(dataset))
    split_tp = parse_split_file(dataset)
    f_info = parse_directory(frame_path, rgb_p, flow_x_p, flow_y_p)

    print('- Writing list files for training/testing')
    for i in range(max(num_split, len(split_tp))):
        lists = build_split_list(split_tp, f_info, i, shuffle)
        open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][0])
        open(os.path.join(out_path, '{}_rgb_val_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][1])
    print('- List files successfully saved to "data/" folder!\n')

else:
    print('"{}" dataset have not been tested yet!'.format(dataset))
