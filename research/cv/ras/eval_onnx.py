"""
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
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import mindspore.ops as ops
from mindspore import Tensor

from src.dataset_test import TrainDataLoader

sys.path.append("../")


# data_url is the directory where the data set is located,
# and there must be two folders, images and gts, under data_url;


parser = argparse.ArgumentParser()
parser.add_argument('--device_target', type=str, default="Ascend", help="Ascend, GPU, CPU")
parser.add_argument('--data_url', type=str)
parser.add_argument('--save_url', type=str)
parser.add_argument('--onnx_file', type=str)

par = parser.parse_args()


def image_loader(imagename):
    image = Image.open(imagename).convert("L")
    return np.array(image)


def Fmeasure(predict_, groundtruth):
    """

    Args:
        predict: predict image
        gt: ground truth

    Returns:
        Calculate F-measure
    """
    sumLabel = 2 * np.mean(predict_)
    if sumLabel > 1:
        sumLabel = 1
    Label3 = predict_ >= sumLabel
    NumRec = np.sum(Label3)
    #LabelAnd = (Label3 is True)
    LabelAnd = Label3
    #NumAnd = np.sum(np.logical_and(LabelAnd, groundtruth))
    gt_t = gt > 0.5
    NumAnd = np.sum(LabelAnd * gt_t)
    num_obj = np.sum(groundtruth)
    if NumAnd == 0:
        p = 0
        r = 0
        FmeasureF = 0
    else:
        p = NumAnd / NumRec
        r = NumAnd / num_obj
        FmeasureF = (1.3 * p * r) / (0.3 * p + r)
    return FmeasureF


def create_session(onnx_checkpoint_path, target_device='GPU'):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(onnx_checkpoint_path, providers=providers)

    input_name = session.get_inputs()[0].name
    return session, input_name


if __name__ == "__main__":
    filename = os.path.join(par.data_url, 'images/')
    gtname = os.path.join(par.data_url, 'gts/')
    save_path = par.save_url
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    testdataloader = TrainDataLoader(filename)

    sess, input_sess = create_session(par.onnx_file, par.device_target)

    Names = []
    for data in os.listdir(filename):
        name = data.split('.')[0]
        Names.append(name)
    Names = sorted(Names)
    i = 0
    sigmoid = ops.Sigmoid()
    for data in testdataloader.dataset.create_dict_iterator(output_numpy=True):
        data, data_org = data["data"], data["data_org"]
        img = sess.run(None, {input_sess: data})[0]
        img = Tensor(img)
        upsample = ops.ResizeBilinear((data_org.shape[1], data_org.shape[2]), align_corners=False)
        img = upsample(img)
        img = sigmoid(img)
        img = img.asnumpy().squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 255
        data_name = Names[i]
        save_path_end = os.path.join(save_path, data_name + '.png')
        cv2.imwrite(save_path_end, img)
        print("---------------  %d OK ----------------" % i)
        i += 1
    print("-------------- EVALUATION END --------------------")
    predictpath = par.save_url

    # calculate F-measure
    gtfiles = sorted([gtname + gt_file for gt_file in os.listdir(gtname)])
    predictfiles = sorted([os.path.join(predictpath, predictfile) for predictfile in os.listdir(predictpath)])

    Fs = []
    for i in range(len(gtfiles)):
        gt = image_loader(gtfiles[i]) / 255
        predict = image_loader(predictfiles[i]) / 255
        fmea = Fmeasure(predict, gt)
        Fs.append(fmea)

    print("Fmeasure is %.3f" % np.mean(Fs))
