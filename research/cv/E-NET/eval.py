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
"""eval Enet"""

import math
import os
from argparse import ArgumentParser

import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.criterion import SoftmaxCrossEntropyLoss
from src.dataset import getCityScapesDataLoader_GeneratorDataset
from src.iou_eval import iouEval
from src.model import Encoder_pred, Enet
from src.util import getBool, getCityLossWeight


def IOU(network_trained, dataloader, num_class, enc):
    """compute IOU"""
    ioueval = iouEval(num_class)
    loss = SoftmaxCrossEntropyLoss(num_class, getCityLossWeight(enc))
    loss_list = []
    network_trained.set_train(False)
    for index, (images, labels) in enumerate(dataloader):
        preds = network_trained(images)
        l = loss(preds, labels)
        loss_list.append(float(str(l)))
        print("step {}/{}: loss:  ".format(index+1, dataloader.get_dataset_size()), l)
        preds = preds.asnumpy().argmax(axis=1).astype(np.int32)
        labels = labels.asnumpy().astype(np.int32)
        ioueval.addBatch(preds, labels)

    mean_iou, iou_class = ioueval.getIoU()
    mean_iou = mean_iou.item()
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_iou, mean_loss, iou_class

def evalNetwork(network, eval_dataloader, ckptPath, encode, num_class=20):
    """load model,eval and save result"""
    if ckptPath is None:
        print("no model checkpoint!")
    elif not os.path.exists(ckptPath):
        print("not exist {}".format(ckptPath))
    else:
        print("load model checkpoint {}!".format(ckptPath))
        param_dict = load_checkpoint(ckptPath)
        load_param_into_net(network, param_dict)

    mean_iou, mean_loss, iou_class = IOU(network, eval_dataloader, num_class, encode)
    with open(ckptPath + ".metric.txt", "w") as file_:
        print("model path", ckptPath, file=file_)
        print("mean_iou", mean_iou, file=file_)
        print("mean_loss", mean_loss, file=file_)
        print("iou_class", iou_class, file=file_)

def listCKPTPath(model_root_path, enc):
    """get all the ckpt path in model_root_path"""
    paths = []
    names = os.listdir(model_root_path)
    for name in names:
        if name.endswith(".ckpt") and name+".metric.txt" not in names:
            if enc and name.startswith("Encoder"):
                ckpt_path = os.path.join(model_root_path, name)
                paths.append(ckpt_path)
            elif not enc and name.startswith("ENet"):
                ckpt_path = os.path.join(model_root_path, name)
                paths.append(ckpt_path)
    return paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--run_distribute', type=str)
    parser.add_argument('--encode', type=str)
    parser.add_argument('--model_root_path', type=str)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device_target', type=str, default='Ascend')

    config = parser.parse_args()
    model_root_path_ = config.model_root_path
    encode_ = getBool(config.encode)
    device_id = config.device_id
    CityScapesRoot = config.data_path
    run_distribute = getBool(config.run_distribute)
    device_target = config.device_target

    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target=device_target)
    context.set_context(device_id=device_id)
    context.set_context(save_graphs=False)

    eval_dataloader_ = getCityScapesDataLoader_GeneratorDataset(CityScapesRoot, "val", 6, \
                                                               encode_, 512, False, False)

    weight_init = "XavierUniform"
    if encode_:
        network_ = Encoder_pred(num_class=20, weight_init=weight_init, train=False)
    else:
        network_ = Enet(num_classes=20, init_conv=weight_init, train=False)

    if not run_distribute:
        if os.path.isdir(model_root_path_):
            paths_ = listCKPTPath(model_root_path_, encode_)
            for path in paths_:
                evalNetwork(network_, eval_dataloader_, path, encode_)
        else:
            evalNetwork(network_, eval_dataloader_, model_root_path_, encode_)
    else:
        rank_id = int(os.environ["RANK_ID"])
        rank_size = int(os.environ["RANK_SIZE"])
        ckpt_files_path = listCKPTPath(model_root_path_, encode_)
        n = math.ceil(len(ckpt_files_path) / rank_size)
        ckpt_files_path = ckpt_files_path[rank_id*n : rank_id*n + n]

        for path in ckpt_files_path:
            evalNetwork(network_, eval_dataloader_, path, encode_)
