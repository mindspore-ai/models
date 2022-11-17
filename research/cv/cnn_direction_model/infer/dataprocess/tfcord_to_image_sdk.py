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

import argparse
import csv

import os
import re
from io import BytesIO
import json
import itertools
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tqdm import tqdm


image_width = 64
image_height = 512

def resize_image(pic):
    color_fill = 255
    scale = image_height / pic.shape[0]
    pic = cv2.resize(pic, None, fx=scale, fy=scale)
    if pic.shape[1] > image_width:
        pic = pic[:, 0:image_width]
    else:
        blank_img = np.zeros((image_height, image_width, 3), np.uint8)
        # fill the image with white
        blank_img.fill(color_fill)
        blank_img[:image_height, :pic.shape[1]] = pic
        pic = blank_img
    data = np.array([pic[...]], np.float32)
    data = data / 127.5 - 1
    return data



FILENAME_PATTERN = re.compile(r'.+-(\d+)-of-(\d+)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that takes tfrecord files and \
                                     extracts all images + labels from it')
    parser.add_argument('tfrecord_dir', default='./data/val', help='path to directory containing tfrecord files')
    parser.add_argument('destination_dir', default='./data', help='path to dir where resulting images shall be saved')
    parser.add_argument('stage', default='train', help='stage of training these files are for [e.g. train]')
    parser.add_argument('char_map', help='path to fsns char map')
    parser.add_argument('destination', help='path to destination gt file')
    parser.add_argument('--max-words', type=int, default=6, help='max words per image')
    parser.add_argument('--min-words', type=int, default=1, help='min words per image')
    parser.add_argument('--max-chars', type=int, default=21, help='max characters per word')
    parser.add_argument('--word-gt', action='store_true', default=False, help='input gt is word level gt')
    parser.add_argument('--blank-label', default='133', help='class number of blank label')

    args = parser.parse_args()

    os.makedirs(args.destination_dir, exist_ok=True)

    tfrecord_files = os.listdir(args.tfrecord_dir)
    tfrecord_files = sorted(tfrecord_files, key=lambda x: int(FILENAME_PATTERN.match(x).group(1)))
    fsns_gt = os.path.join(args.destination_dir, '{}.csv'.format(args.stage))
    with open(fsns_gt, 'w') as label_file:
        writer = csv.writer(label_file, delimiter='\t')
        idx_tmp = 0
        for tfrecord_file in tfrecord_files:
            tfrecord_filename = os.path.join(args.tfrecord_dir, tfrecord_file)

            file_id = '00000'
            dest_dir = os.path.join(args.destination_dir, args.stage, file_id)
            os.makedirs(dest_dir, exist_ok=True)

            record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=tfrecord_filename)

            for idx, string_record in enumerate(record_iterator):
                idx_tmp += 1
                example = tf.train.Example()
                example.ParseFromString(string_record)

                labels = example.features.feature['image/class'].int64_list.value
                img_string = example.features.feature['image/encoded'].bytes_list.value[0]

                image = Image.open(BytesIO(img_string))
                img = np.array(image)

                img = img[:150, :150, :]
                im = Image.fromarray(img)
                if np.random.rand() > 0.5:
                    file_name = os.path.join(dest_dir, '{}_1.jpg'.format(idx_tmp))
                    im.save(file_name)

                    label_file_data = [os.path.join(args.stage, file_id, '{}_1.jpg'.format(idx_tmp))]
                    label_file_data.extend(labels)
                    writer.writerow(label_file_data)
                else:
                    # rot image
                    img_rotate = np.rot90(img)
                    img = np.rot90(img_rotate)
                    img_rot_string = img.tobytes()
                    im = Image.fromarray(img)
                    file_name = os.path.join(dest_dir, '{}_0.jpg'.format(idx_tmp))
                    im.save(file_name)

                    label_file_data = [os.path.join(args.stage, file_id, '{}_0.jpg'.format(idx_tmp))]
                    label_file_data.extend(labels)
                    writer.writerow(label_file_data)
                print("recovered {:0>6} files".format(idx), end='\r')

    with open(args.char_map) as c_map:
        char_map = json.load(c_map)
        reverse_char_map = {v: k for k, v in char_map.items()}

    with open(fsns_gt) as fsns_gt_f:
        reader = csv.reader(fsns_gt_f, delimiter='\t')
        lines = [l for l in reader]

    text_lines = []
    for line in tqdm(lines):
        text = ''.join(map(lambda x: chr(char_map[x]), line[1:]))
        if args.word_gt:
            text = text.split(chr(char_map[args.blank_label]))
            text = filter(lambda x: x != [], text)
        else:
            text = text.strip(chr(char_map[args.blank_label]))
            text = text.split()

        words = []
        for t in text:
            t = list(map(lambda x: reverse_char_map[ord(x)], t))
            t.extend([args.blank_label] * (args.max_chars - len(t)))
            words.append(t)

        if line == []:
            continue

        words.extend([[args.blank_label] * args.max_chars for _ in range(args.max_words - len(words))])

        text_lines.append([line[0]] + list(itertools.chain(*words)))

    with open(args.destination, 'w') as dest:
        writer = csv.writer(dest, delimiter='\t')
        writer.writerow([args.max_words, args.max_chars])
        writer.writerows(text_lines)
