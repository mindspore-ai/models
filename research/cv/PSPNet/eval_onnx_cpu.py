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
""" ONNX EVALUATE CPU"""
import os
import time
import logging
import argparse
import cv2
import numpy
from src.dataset import pt_dataset
from src.dataset import pt_transform as trans
import src.utils.functions_args as fa
from src.utils.p_util import AverageMeter, intersectionAndUnion, check_makedirs, colorize
import mindspore.numpy as np
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore import context
import mindspore.nn as nn
import mindspore.ops as ops
import onnxruntime

cv2.ocl.setUseOpenCL(False)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU",
                    save_graphs=False)

def get_logger():
    """ logger """
    logger_name = "main-logger"
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    return log

# Due to Onnx-gpu can not support PSPNet Inference on GPU platform, this function only provide CPUExecutionProvider.
def getonnxmodel():
    model = onnxruntime.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
    return model

def get_Config():
    config_parser = argparse.ArgumentParser(description='MindSpore Semantic Segmentation')
    config_parser.add_argument('--config', type=str, required=True, help='config file')
    config_parser.add_argument('opts', help='see ./src/config/voc2012_pspnet50.yaml for all options', default=None,
                               nargs=argparse.REMAINDER)
    args_ = config_parser.parse_args()
    assert args_.config is not None
    cfg = fa.load_cfg_from_cfg_file(args_.config)
    if args_.opts is not None:
        cfg = fa.merge_cfg_from_list(cfg, args_.opts)
    return cfg

def main():
    """ The main function of the evaluate process """
    logger.info("=> Load PSPNet ...")
    logger.info("%s: class num:%s", args.prefix, args.classes)
    value_scale = 255
    m = [0.485, 0.456, 0.406]
    m = [item * value_scale for item in m]
    s = [0.229, 0.224, 0.225]
    s = [item * value_scale for item in s]
    gray_folder = os.path.join(args.result_path, 'gray')
    color_folder = os.path.join(args.result_path, 'color')

    test_transform = trans.Compose([trans.Normalize(mean=m, std=s, is_train=False)])
    test_data = pt_dataset.SemData(
        split='val', data_root=args.data_root,
        data_list=args.val_list,
        transform=test_transform)

    test_loader = ds.GeneratorDataset(test_data, column_names=["data", "label"],
                                      shuffle=False)
    test_loader.batch(1)
    colors = numpy.loadtxt(args.color_txt).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.name_txt)]
    model = getonnxmodel()

    test(test_loader, test_data.data_list, model, args.classes, m, s, args.base_size, args.test_h,
         args.test_w, args.scales, gray_folder, color_folder, colors)
    if args.split != 'test':
        calculate_acc(test_data.data_list, gray_folder, args.classes, names, colors)


def net_process(model, image, mean, std=None, flip=False):
    """ Give the input to the model"""
    transpose = ops.Transpose()
    input_ = transpose(image, (2, 0, 1))  # (473, 473, 3) -> (3, 473, 473)
    mean = np.array(mean)
    std = np.array(std)
    if std is None:
        input_ = input_ - mean[:, None, None]
    else:
        input_ = (input_ - mean[:, None, None]) / std[:, None, None]

    expand_dim = ops.ExpandDims()
    input_ = expand_dim(input_, 0)
    if flip:
        flip_input = np.flip(input_, axis=3)
        concat = ops.Concat(axis=0)
        input_ = concat((input_, flip_input))
    input_ = input_.asnumpy()
    inputs = {model.get_inputs()[0].name: input_}
    output = model.run(None, inputs)
    output = Tensor(output)
    _, _, h_i, w_i = input_.shape
    _, _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        bi_linear = nn.ResizeBilinear()
        output = bi_linear(output, size=(h_i, w_i), align_corners=True)

    softmax = nn.Softmax(axis=2)
    output = softmax(output)
    if flip:
        output = (output[0] + np.flip(output[1], axis=2)) / 2
    else:
        output = output[0][0]
    output = transpose(output, (1, 2, 0))  # Tensor
    output = output.asnumpy()
    return output


def scale_proc(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
    """ Process input size """
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = image.shape
    image = Tensor.from_numpy(image)
    stride_h = int(numpy.ceil(crop_h * stride_rate))
    stride_w = int(numpy.ceil(crop_w * stride_rate))
    g_h = int(numpy.ceil(float(new_h - crop_h) / stride_h) + 1)
    g_w = int(numpy.ceil(float(new_w - crop_w) / stride_w) + 1)
    pred_crop = numpy.zeros((new_h, new_w, classes), dtype=float)
    count_crop = numpy.zeros((new_h, new_w), dtype=float)
    for idh in range(0, g_h):
        for idw in range(0, g_w):
            s_h = idh * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = idw * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            pred_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    pred_crop /= numpy.expand_dims(count_crop, 2)
    pred_crop = pred_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    pred = cv2.resize(pred_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred


def test(test_loader, data_list, model, classes, m, s, base_size, crop_h, crop_w, scales, gray_folder,
         color_folder, colors):
    """ Generate evaluate image """
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    data_num = len(data_list)
    scales_num = len(scales)
    for index, (input_, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input_ = input_.asnumpy()
        image = numpy.transpose(input_, (1, 2, 0))
        height, weight, _ = image.shape
        pred = numpy.zeros((height, weight, classes), dtype=float)
        for ratio in scales:
            long_size = round(ratio * base_size)
            new_h = long_size
            new_w = long_size
            if height > weight:
                new_w = round(long_size / float(height) * weight)
            else:
                new_h = round(long_size / float(weight) * height)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pred += scale_proc(model, image_scale, classes, crop_h, crop_w, height, weight, m, s)
        pred = pred / scales_num
        pred = numpy.argmax(pred, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((index + 1) % 10 == 0) or (index + 1 == data_num):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(index + 1, data_num,
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = numpy.uint8(pred)
        color = colorize(gray, colors)
        image_path, _ = data_list[index]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_img = os.path.join(gray_folder, image_name + '.png')
        color_img = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_img, gray)
        color.save(color_img)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

def convert_label(label, colors):
    """Convert classification ids in labels."""
    mask_map = numpy.zeros((label.shape[0], label.shape[1]))
    for i in range(len(label)):
        for j in range(len(label[i])):
            if colors.count(label[i][j].tolist()):
                mask_map[i][j] = colors.index(label[i][j].tolist())
    import mindspore
    a = Tensor(mask_map, dtype=mindspore.uint8)
    mask_map = a.asnumpy()
    return mask_map

def calculate_acc(data_list, pred_folder, classes, names, colors):
    """ Calculation evaluating indicator """
    colors = colors.tolist()
    overlap_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for i, (image_path, label_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        if args.prefix != "ADE":
            target = cv2.imread(label_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            anno_label = convert_label(target, colors)

        if args.prefix == 'ADE':
            anno_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            anno_label -= 1
        overlap, union, label = intersectionAndUnion(pred, anno_label, args.classes)
        overlap_meter.update(overlap)
        union_meter.update(union)
        target_meter.update(label)
        accuracy = sum(overlap_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info(
            'Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name + '.png',
                                                                        accuracy))
    iou_class = overlap_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = overlap_meter.sum / (target_meter.sum + 1e-10)
    mIoU = numpy.mean(iou_class)
    mAcc = numpy.mean(accuracy_class)
    allAcc = sum(overlap_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))


if __name__ == '__main__':
    args = get_Config()
    logger = get_logger()
    main()
