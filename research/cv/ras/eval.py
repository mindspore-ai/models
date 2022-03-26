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

import mindspore as ms
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net

from src.model import BoneModel
from src.dataset_test import TrainDataLoader

sys.path.append("../")


# data_url is the directory where the data set is located,
# and there must be two folders, images and gts, under data_url;

# If inferring on modelarts, there are two zip compressed files named after images and gts under data_url,
# and there are only these two files




parser = argparse.ArgumentParser()
parser.add_argument('--is_modelarts', type=str, default="NO")
parser.add_argument('--device_target', type=str, default="Ascend", help="Ascend, GPU, CPU")
parser.add_argument('--device_id', type=int, default=5, help='Number of device')
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--pre_model', type=str)

par = parser.parse_args()


device_target = par.device_target
if par.is_modelarts == "YES":
    device_id = int(os.getenv("DEVICE_ID"))
else:
    device_id = int(par.device_id)

ms.context.set_context(device_target=device_target, device_id=device_id)

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




if __name__ == "__main__":
    if par.is_modelarts == "YES":
        data_true_path = par.data_url
        pre_model_true_path = par.pre_model
        result_path = par.train_url
        model_true_path = par.model_path
        import moxing as mox

        test_out = '/cache/test_output/'
        local_data_path = '/cache/test/'
        os.system("mkdir {0}".format(test_out))
        os.system("mkdir {0}".format(local_data_path))
        image_name = "images.zip"
        gt_name = "gts.zip"
        mox.file.copy_parallel(src_url=data_true_path, dst_url=local_data_path)
        mox.file.copy_parallel(src_url=pre_model_true_path, dst_url=local_data_path)
        mox.file.copy_parallel(src_url=model_true_path, dst_url=local_data_path)
        zip_command1 = "unzip -o -q %s -d %s" % (local_data_path + image_name, local_data_path)
        zip_command2 = "unzip -o -q %s -d %s" % (local_data_path + gt_name, local_data_path)
        os.system(zip_command1)
        os.system(zip_command2)
        print("unzip success")

        filename = os.path.join(local_data_path, "images/")
        gtname = os.path.join(local_data_path, 'gts/')
        pre_model_path = os.path.join(local_data_path, pre_model_true_path.split("/")[-1])
        trained_model_path = os.path.join(local_data_path, model_true_path.split("/")[-1])
    else:
        filename = os.path.join(par.data_url, 'images/')
        gtname = os.path.join(par.data_url, 'gts/')
        pre_model_path = par.pre_model
        trained_model_path = par.model_path
        save_path = par.train_url
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    testdataloader = TrainDataLoader(filename)

    model = BoneModel(device_target, pre_model_path)
    param_dict = load_checkpoint(trained_model_path)
    load_param_into_net(model, param_dict)

    Names = []
    for data in os.listdir(filename):
        name = data.split('.')[0]
        Names.append(name)
    Names = sorted(Names)
    i = 0
    sigmoid = ops.Sigmoid()
    for data in testdataloader.dataset.create_dict_iterator():
        data, data_org = data["data"], data["data_org"]
        img, _, _, _, _ = model(data)
        upsample = ops.ResizeBilinear((data_org.shape[1], data_org.shape[2]), align_corners=False)
        img = upsample(img)
        img = sigmoid(img)
        img = img.asnumpy().squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 255
        data_name = Names[i]
        if par.is_modelarts == "NO":
            save_path_end = os.path.join(save_path, data_name + '.png')
        else:
            save_path_end = os.path.join(test_out, data_name + '.png')
        cv2.imwrite(save_path_end, img)
        print("---------------  %d OK ----------------" % i)
        i += 1
    print("-------------- EVALUATION END --------------------")
    if par.is_modelarts == "YES":
        predictpath = test_out
        mox.file.copy_parallel(src_url=test_out, dst_url=result_path)
    else:
        predictpath = par.train_url

    #calculate F-measure
    gtfiles = sorted([gtname + gt_file for gt_file in os.listdir(gtname)])
    predictfiles = sorted([os.path.join(predictpath, predictfile) for predictfile in os.listdir(predictpath)])

    Fs = []
    for i in range(len(gtfiles)):
        gt = image_loader(gtfiles[i]) / 255
        predict = image_loader(predictfiles[i]) / 255
        fmea = Fmeasure(predict, gt)
        Fs.append(fmea)

    print("Fmeasure is %.3f" % np.mean(Fs))
