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

import os
import sys
import json
import cv2


def unzip_rawrarfile(dataset_name, root_dir_path):
    os.chdir(root_dir_path)
    if dataset_name == 'HMDB51':
        # unrar split file
        os.system(command='rar x test_train_splits.rar')
        os.system(command='mv ./testTrainMulti_7030_splits ./hmdb-51_txt')
        # unrar video file
        os.system(command='rar x hmdb51_org.rar  hmdb-51/ ')
        os.system(command='mkdir hmdb-51_video/')
        os.chdir('hmdb-51_video/')
        os.system(command='ls ../hmdb-51/*.rar | xargs -n1 rar x')
        os.system(command='rm -rf ../hmdb-51/')

    elif dataset_name == 'UCF101':
        # unrar split file
        os.system(command='unzip UCF101TrainTestSplits-RecognitionTask.zip')
        os.system(command='mv ./ucfTrainTestlist ./UCF-101_txt')
        # unrar video file
        os.system(command='rar x UCF101.rar ')
        os.system(command='mv UCF-101/ UCF-101_video/')

    else:
        print('Dataset {} is not surpported !'.format(dataset_name))
        raise InterruptedError


def video_2_img(dataset_name, root_dir_path):
    if dataset_name == 'HMDB51':
        img_dir_name = 'hmdb-51_img'
        video_dir_name = 'hmdb-51_video'
    elif dataset_name == 'UCF101':
        img_dir_name = 'UCF-101_img'
        video_dir_name = 'UCF-101_video'
    else:
        print('Dataset {} is not surpported !'.format(dataset_name))
        raise InterruptedError
    img_dir = os.path.join(root_dir_path, img_dir_name)
    video_dir = os.path.join(root_dir_path, video_dir_name)
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        output_dir_ = os.path.join(img_dir, file)
        for video in os.listdir(file_path):
            process_video(video, file, output_dir_, video_dir)

    print('DataSet Preprocessing finished.')


def process_video(video, action_name, save_dir, video_dir):
    # Initialize a VideoCapture object to read video data into a numpy array
    video_filename = video.split('.')[0]
    os.makedirs(os.path.join(save_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(os.path.join(video_dir, action_name, video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make sure splited video has at least 16 frames
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1

    # Release the VideoCapture once it is no longer needed
    capture.release()

    print(action_name, video, count, i, frame_count)


def gen_json_HMDB51(root_dir_path, flod=1):
    json_root = os.path.join(root_dir_path, 'hmdb-51_json')
    txt_root = os.path.join(root_dir_path, 'hmdb-51_txt')
    img_path = os.path.join(root_dir_path, 'hmdb-51_img')
    os.makedirs(json_root, exist_ok=True)
    train_list, test_list, val_list = [], [], []
    label_dict = {'brush_hair': 0, 'cartwheel': 1,
                  'catch': 2, 'chew': 3,
                  'clap': 4, 'climb_stairs': 5,
                  'climb': 6, 'dive': 7,
                  'draw_sword': 8, 'dribble': 9,
                  'drink': 10, 'eat': 11,
                  'fall_floor': 12, 'fencing': 13,
                  'flic_flac': 14, 'golf': 15,
                  'handstand': 16, 'hit': 17,
                  'hug': 18, 'jump': 19,
                  'kick_ball': 20, 'kick': 21,
                  'kiss': 22, 'laugh': 23,
                  'pick': 24, 'pour': 25,
                  'pullup': 26, 'punch': 27,
                  'push': 28, 'pushup': 29,
                  'ride_bike': 30, 'ride_horse': 31,
                  'run': 32, 'shake_hands': 33,
                  'shoot_ball': 34, 'shoot_bow': 35,
                  'shoot_gun': 36, 'sit': 37,
                  'situp': 38, 'smile': 39,
                  'smoke': 40, 'somersault': 41,
                  'stand': 42, 'swing_baseball': 43,
                  'sword_exercise': 44, 'sword': 45,
                  'talk': 46, 'throw': 47,
                  'turn': 48, 'walk': 49,
                  'wave': 50}
    for txt_file_name in os.listdir(txt_root):
        txt_file_name_ = txt_file_name.split('_test_')[0]
        if 'split{}.txt'.format(flod) in txt_file_name:
            with open(os.path.join(txt_root, txt_file_name), 'r') as txt_file:
                for info_ in txt_file.readlines():
                    info = info_.strip().split(' ')
                    if len(info) == 2:
                        if info[1] == '1':
                            train_list.append((txt_file_name_, info[0]))
                        elif info[1] == '2':
                            test_list.append((txt_file_name_, info[0]))
                        elif info[1] == '0':
                            val_list.append((txt_file_name_, info[0]))
                    else:
                        print(info)
                        continue
    write_json(train_list, os.path.join(json_root, 'train.json'), img_path, label_dict)
    write_json(test_list, os.path.join(json_root, 'test.json'), img_path, label_dict)
    write_json(val_list, os.path.join(json_root, 'val.json'), img_path, label_dict)


def gen_json_UCF101(root_dir_path, flod=1):
    json_root = os.path.join(root_dir_path, 'UCF-101_json')
    txt_root = os.path.join(root_dir_path, 'UCF-101_txt')
    img_path = os.path.join(root_dir_path, 'UCF-101_img')
    os.makedirs(json_root, exist_ok=True)
    train_list, test_list = [], []
    label_dict = {}
    with open(os.path.join(txt_root, 'classInd.txt'), 'r') as class_file:
        for index, line in enumerate(class_file.readlines()):
            if line.strip():
                label_dict[line.strip().split(' ')[1]] = index

    with open(os.path.join(txt_root, 'trainlist0{}.txt'.format(flod)), 'r') as file:
        for line in file.readlines():
            if line.strip():
                line = line.strip().split(' ')[0]
                train_list.append(line.split('/'))

    with open(os.path.join(txt_root, 'testlist0{}.txt'.format(flod)), 'r') as file:
        for line in file.readlines():
            if line.strip():
                test_list.append(line.strip().split('/'))
    write_json(train_list, os.path.join(json_root, 'train.json'), img_path, label_dict)
    write_json(test_list, os.path.join(json_root, 'test.json'), img_path, label_dict)


def write_json(video_list, dest_path, img_path, label_dict):
    dest_data = []
    for action, video in video_list:
        video_images = sorted(os.listdir(os.path.join(img_path, action, video.replace('.avi', ''))))
        samples = [os.path.join(img_path, action, video.replace('.avi', ''), video_image)
                   for video_image in video_images]

        dest_data_lvl1 = {'frames': []}
        for frame in samples:
            dest_data_lvl1['frames'].append(
                {'img_path': os.path.split(frame)[1], 'actions': [{'action_class': label_dict[action]}]})

        dest_data_lvl1['base_path'] = os.path.join(action, video.replace('.avi', ''))
        dest_data.append(dest_data_lvl1)

    with open(dest_path, 'w') as outfile:
        json.dump(dest_data, outfile, indent=4)


def gen_json(dataset, root_dir_path, split_flod):
    if dataset == 'HMDB51':
        gen_json_HMDB51(root_dir_path, split_flod)
    elif dataset == 'UCF101':
        gen_json_UCF101(root_dir_path, split_flod)
    else:
        print('Dataset {} is not surpported !'.format(dataset))
        raise InterruptedError
    print('Generating Training and Test JSON Files OK !')


def data_preprocess(dataset, root_dir_path, split_flod=1):
    if os.path.isfile(root_dir_path):
        root_dir_path = os.path.dirname(root_dir_path)
    unzip_rawrarfile(dataset, root_dir_path)
    video_2_img(dataset, root_dir_path)
    gen_json(dataset, root_dir_path, split_flod)


if __name__ == "__main__":
    dataset_name_ = sys.argv[1]  # Dataset name
    root_dir_path_ = sys.argv[2]  # Dataset rar file path
    split_flod_ = sys.argv[3]  # Dataset rar file path
    data_preprocess(dataset_name_, root_dir_path_, split_flod_)
