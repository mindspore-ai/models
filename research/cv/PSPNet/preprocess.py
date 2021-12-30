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
""" PREPROCESS BEFORE 310 INFER """
import os
import logging
import argparse
import cv2
import numpy
from src.dataset import pt_dataset, pt_transform
import src.utils.functions_args as fa

cv2.ocl.setUseOpenCL(False)
Small_block_name = []
aux_inputs_name = []

parser = argparse.ArgumentParser(description='MindSpore Semantic Segmentation')
parser.add_argument('--config', type=str, required=True, default=None, help='config file')
parser.add_argument('--save_path', type=str, required=True, default=None, help='save the preprocess file')
parser.add_argument('--data_path', type=str, required=True, default=None, help='data path')
parser.add_argument('opts', help='see voc2012_pspnet50.yaml/ade20k_pspnet50.yaml for all options', default=None,
                    nargs=argparse.REMAINDER)
args_ = parser.parse_args()
cfg = fa.load_cfg_from_cfg_file(args_.config)


def get_logger():
    """ logger """
    logger_name = "main-logger"
    logger_ = logging.getLogger(logger_name)
    logger_.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_.addHandler(handler)
    return logger_


def check(local_args):
    """ check args """
    assert local_args.classes > 1
    assert local_args.zoom_factor in [1, 2, 4, 8]
    assert local_args.split in ['train', 'val', 'test']
    if local_args.arch == 'psp':
        assert (local_args.train_h - 1) % 8 == 0 and (local_args.train_w - 1) % 8 == 0
    else:
        raise Exception('architecture not supported {} yet'.format(local_args.arch))


def main():
    """ The main function of the preprocess """
    check(cfg)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    test_transform = pt_transform.Compose([pt_transform.Normalize(mean=mean, std=std, is_train=False)])
    test_data = pt_dataset.SemData(
        split='val', data_root=args_.data_path,
        data_list=args_.data_path + 'val_list.txt',
        transform=test_transform)

    split_image(test_data, mean, std, cfg.base_size, cfg.test_h, cfg.test_w, cfg.scales)


def before_net(image, mean, std=None, flip=True):
    """ Give the input to the model"""
    input_ = numpy.transpose(image, (2, 0, 1))  # (473, 473, 3) -> (3, 473, 473)
    mean = numpy.array(mean)
    std = numpy.array(std)
    if std is None:
        input_ = input_ - mean[:, None, None]
    else:
        input_ = (input_ - mean[:, None, None]) / std[:, None, None]

    input_ = numpy.expand_dims(input_, 0)
    if flip:
        flip_input = numpy.flip(input_, axis=[3])
        input_ = numpy.concatenate((input_, flip_input), axis=0)

    return input_


def process_image(image, image_idx, crop_h, crop_w, mean, std=None, stride_rate=2 / 3):
    """ Process input size """
    original_h, original_w, _ = image.shape
    pad_w = max(0, crop_w - original_w)
    pad_h = max(0, crop_h - original_h)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    file_name = post_save + '/' + str(image_idx)
    idx = 0
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(numpy.ceil(crop_h * stride_rate))
    stride_w = int(numpy.ceil(crop_w * stride_rate))
    grid_h = int(numpy.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(numpy.ceil(float(new_w - crop_w) / stride_w) + 1)
    count_crop = numpy.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            idx += 1
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            image_crop = before_net(image_crop, mean, std)
            image_crop = image_crop.astype(numpy.float32)
            image_crop.tofile(file_name + '-' + str(idx) + '.bin')
            Small_block_name.append(file_name + '-' + str(idx) + '.bin')
    count_crop.tofile(count_save + '/' + str(image_idx) + '.bin')
    aux_inputs_name.append(count_save + '/' + str(image_idx) + '.bin')


def split_image(test_loader, mean, std, base_size, crop_h, crop_w, scales):
    """ Get input image with fixed size"""
    for i, (input_, _) in enumerate(test_loader):
        print('PROCESS IMAGE ', i + 1)
        image = numpy.transpose(input_, (1, 2, 0))
        h, w, _ = image.shape
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            process_image(image_scale, i + 1, crop_h, crop_w, mean, std)


if __name__ == '__main__':
    post_save = os.path.join(args_.save_path, 'inputs')
    count_save = os.path.join(args_.save_path, 'aux_inputs')
    os.mkdir(post_save)
    os.mkdir(count_save)
    main()
    for bin_name in Small_block_name:
        f = open(args_.save_path + 'inputs.txt', 'a')
        f.write(bin_name)
        f.write('\n')
        f.close()

    for aux_name in aux_inputs_name:
        f = open(args_.save_path + 'aux_inputs.txt', 'a')
        f.write(aux_name)
        f.write('\n')
        f.close()
