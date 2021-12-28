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
""" POSTPROCESS FOR 310 INFER """
import os
import logging
import argparse
import cv2
import numpy as np
from src.dataset import pt_dataset, pt_transform
import src.utils.functions_args as fa
from src.utils.p_util import AverageMeter, intersectionAndUnion, check_makedirs, colorize
import mindspore
from mindspore import Tensor
from mindspore import context
import mindspore.nn as nn

cv2.ocl.setUseOpenCL(False)
context.set_context(device_target="CPU")

parser = argparse.ArgumentParser(description='MindSpore Semantic Segmentation')
parser.add_argument('--config', type=str, required=True, default=None, help='config file')
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
    """ The main function of the postprocess """
    check(cfg)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    gray_folder = os.path.join('./postprocess_Result/', 'gray')
    color_folder = os.path.join('./postprocess_Result/', 'color')

    test_transform = pt_transform.Compose([pt_transform.Normalize(mean=mean, std=std, is_train=False)])
    test_data = pt_dataset.SemData(
        split='val', data_root=args_.data_path,
        data_list=args_.data_path + 'val_list.txt',
        transform=test_transform)
    color_name_path = os.path.dirname(args_.config)
    if cfg.prefix == 'voc':
        color_file = color_name_path + '/voc2012/voc2012_colors.txt'
        name_file = color_name_path + '/voc2012/voc2012_names.txt'
    else:
        color_file = color_name_path + '/ade20k/ade20k_colors.txt'
        name_file = color_name_path + '/ade20k/ade20k_names.txt'
    colors = np.loadtxt(color_file).astype('uint8')
    names = [line.rstrip('\n') for line in open(name_file)]

    merge_blocks(test_data, test_data.data_list, cfg.classes, mean, cfg.base_size, cfg.test_h,
                 cfg.test_w, cfg.scales, gray_folder, color_folder, colors)
    acc(test_data.data_list, gray_folder, cfg.classes, names)


def bin_process(image, image_idx, flip=True):
    """ Process Bin File """
    _, _, h_i, w_i = (2, 3, 473, 473)
    _, _, h_o, w_o = dims[image_idx]
    image = np.resize(image, dims[image_idx])
    image = Tensor.from_numpy(image)
    image = image.astype(mindspore.float32)

    if (h_o != h_i) or (w_o != w_i):
        bi_linear = nn.ResizeBilinear()
        image = bi_linear(image, size=(h_i, w_i), align_corners=True)
    softmax = nn.Softmax(axis=1)
    output = softmax(image)
    output = output.asnumpy()

    if flip:
        flip_ = np.flip(output[1], axis=[2])
        output = (output[0] + flip_) / 2
    else:
        output = output[0]
    output = np.transpose(output, (1, 2, 0))
    return output


def merge_aux(image, image_idx, classes, crop_h, crop_w, h, w, mean, stride_rate=2 / 3):
    """ Merge Helper """
    global file_index
    o_h, o_w, _ = image.shape
    pad_w = max(crop_w - o_w, 0)
    pad_h = max(crop_h - o_h, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.fromfile(aux_inputs_file[image_idx], dtype=np.int)
    count_crop = np.resize(count_crop, (new_h, new_w))
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            print("PROCESS IMAGE ", image_idx + 1, " Bin file is ", bin_file[file_index])
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = np.fromfile(bin_file[file_index], dtype=np.float32)
            prediction_crop[s_h:e_h, s_w:e_w, :] += bin_process(image_crop, image_idx)
            file_index += 1
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + o_h, pad_w_half:pad_w_half + o_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def merge_blocks(test_loader, data_list, classes, mean, base_size, crop_h, crop_w, scales, gray_folder,
                 color_folder, colors):
    """ Generate evaluate image """
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    for i, (input_, _) in enumerate(test_loader):
        input_ = np.transpose(input_, (1, 2, 0))
        h, w, _ = input_.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            image_scale = cv2.resize(input_, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += merge_aux(image_scale, i, classes, crop_h, crop_w, h, w, mean)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)

        if ((i + 1) % 10 == 0) or (i + 1 == len(data_list)):
            print('Test: [{}/{}] '.format(i + 1, len(data_list)))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def acc(data_list, pred_folder, classes, names):
    """ Calculate The Accuracy Of 310 Model """
    target_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    logger.info('>>>>>>>>>>>>>>>> Calculate The Accuracy Of Each Predicted Image >>>>>>>>>>>>>>>>')
    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if cfg.prefix != 'voc':
            target -= 1
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info(
            'Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name + '.png',
                                                                        accuracy))

    logger.info('>>>>>>>>>>>>>>>> End The Accuracy Calculation For Each Predicted Image >>>>>>>>>>>>>>>>')
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))


def read_txt(file_path, for_int=False):
    """ Txt File Read Helper"""
    lines = []
    with open(file_path, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            if for_int:
                line = int(line)
            lines.append(line)
    return lines


if __name__ == '__main__':
    logger = get_logger()
    file_index = 0
    bin_file = read_txt('./result_Files/outputs.txt')
    aux_inputs_file = read_txt('./preprocess_Result/aux_inputs.txt')
    dims = read_txt('./result_Files/dims.txt', for_int=True)
    dims = np.resize(dims, (len(aux_inputs_file), 4))
    main()
