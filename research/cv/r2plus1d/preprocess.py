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
'''preprocess'''
import os
import numbers
import argparse

import cv2
import numpy as np

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_path", type=str, default="./dataset_part", help="root path of images")
    parser.add_argument('--output_path', type=str, default="./outputs", help='output_path')
    args_opt = parser.parse_args()
    return args_opt

def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

def resize_clip(clip, size, interpolation='bilinear'):
    '''
    resize the clip
    '''
    assert isinstance(clip[0], np.ndarray)
    if isinstance(size, numbers.Number):
        im_h, im_w, _ = clip[0].shape
        # Min spatial dim already matches minimal size
        if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                               and im_h == size):
            return clip
        new_h, new_w = get_resize_sizes(im_h, im_w, size)
        size = (new_w, new_h)
    else:
        size = size[0], size[1]
    if interpolation == 'bilinear':
        np_inter = cv2.INTER_LINEAR
    else:
        np_inter = cv2.INTER_NEAREST
    scaled = [
        cv2.resize(img, size, interpolation=np_inter) for img in clip
    ]
    return scaled


def normalize(buffer, mean, std):
    for i in range(3):
        buffer[i] = (buffer[i] - mean[i]) / std[i]
    return buffer

def center_crop(clip, size):
    """
    center_crop
    """
    assert isinstance(clip[0], np.ndarray)
    if isinstance(size, numbers.Number):
        size = (size, size)
    h, w = size
    im_h, im_w, _ = clip[0].shape

    if w > im_w or h > im_h:
        error_msg = (
            'Initial image size should be larger then '
            'cropped size but got cropped sizes : ({w}, {h}) while '
            'initial image is ({im_w}, {im_h})'.format(
                im_w=im_w, im_h=im_h, w=w, h=h))
        raise ValueError(error_msg)

    x1 = int(round((im_w - w) / 2.))
    y1 = int(round((im_h - h) / 2.))
    cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]

    return cropped

def loadvideo_decord(sample, sample_rate_scale=1):
    '''loadvideo_decord'''
    frames = sorted([os.path.join(sample, img) for img in os.listdir(sample)])
    frame_count = len(frames)
    frame_list = np.empty((frame_count, 128, 171, 3), np.dtype('float32'))
    for i, frame_name in enumerate(frames):
        frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        frame_list[i] = frame

    # handle temporal segments
    frame_sample_rate = 2
    converted_len = 16 * frame_sample_rate
    seg_len = frame_count

    all_index = []
    if seg_len <= converted_len:
        index = np.linspace(0, seg_len, num=seg_len // frame_sample_rate)
        index = np.concatenate((index, np.ones(16 - seg_len // frame_sample_rate) * seg_len))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        str_idx = end_idx - converted_len
        index = np.linspace(str_idx, end_idx, num=16)
        index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
    all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    buffer = frame_list[all_index]
    return buffer

def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for index, category in enumerate(sorted(os.listdir(os.path.join(args.dataset_root_path, "val")))):
        print("processing category:", category)
        for video in sorted(os.listdir(os.path.join(args.dataset_root_path, "val", category))):
            buffer = loadvideo_decord(os.path.join(args.dataset_root_path, "val", category, video))
            buffer = resize_clip(buffer, 128, "bilinear")
            buffer = center_crop(buffer, size=(112, 112))

            buffer = np.array(buffer).transpose((3, 0, 1, 2)) / 255.0
            buffer = normalize(buffer, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            buffer = buffer.astype(np.float32)

            filename = str(index) + "_" + category + video.split('.')[0]    # get the name of image file
            buffer.tofile(os.path.join(args.output_path, filename+'.bin'))

if __name__ == '__main__':
    np.random.seed(1)
    main()
