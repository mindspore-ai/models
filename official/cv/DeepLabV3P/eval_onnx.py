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
"""Eval deeplabv3plus exported to ONNX"""
import os
import argparse
import numpy as np
import cv2
from eval import pre_process, cal_hist
import onnxruntime


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser('MindSpore DeepLabV3+ eval')
    parser.add_argument('--file_name', type=str, default='deeplabv3plus8.onnx', help='root path of deeplabv3+.onnx')
    parser.add_argument('--data_root', type=str, default='LabV3+Data', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='LabV3+Data/voc_val_lst.txt', help='list of val data')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675], help='image mean')
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')

    # device info
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id')

    args, _ = parser.parse_known_args()
    return args


def create_session(checkpoint_path, target_device):
    """Load ONNX model and create ORT session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU'")
    return onnxruntime.InferenceSession(checkpoint_path, providers=providers)


def eval_batch(args, eval_net, img_lst, crop_size=513, flip=True):
    """Run eval on batch"""
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])
    [input_name] = [x.name for x in eval_net.get_inputs()]
    batch_img = np.ascontiguousarray(batch_img)
    [net_out] = eval_net.run(None, {input_name: batch_img})
    print([net_out])
    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        [net_out_flip] = eval_net.run(None, {input_name: batch_img})
        net_out += net_out_flip[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=513, flip=True):
    """Run eval on batch with different scales"""
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]
    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def net_eval():
    """net_eval"""
    args = parse_args()
    scales_list = [1]
    flip = False
    # data list，
    with open(args.data_lst, encoding='utf-8') as f:
        img_lst = f.readlines()
    session = create_session(args.file_name, args.device_target)
    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    image_num = 0
    for i, line in enumerate(img_lst):
        img_path, msk_path = line.strip().split(' ')
        img_path = os.path.join(args.data_root, img_path)
        msk_path = os.path.join(args.data_root, msk_path)
        img_ = cv2.imread(img_path)
        msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(args, session, batch_img_lst, scales=scales_list,
                                          base_crop_size=args.crop_size, flip=flip)
            for mi in range(args.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            print(f'processed {i + 1} images')
        image_num = i

    if bi > 0:
        batch_res = eval_batch_scales(args, session, batch_img_lst, scales=scales_list,
                                      base_crop_size=args.crop_size, flip=flip)
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        print(f'processed {image_num + 1} images')

    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))


if __name__ == '__main__':
    net_eval()
