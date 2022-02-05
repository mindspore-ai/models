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
import math
from argparse import ArgumentParser
import numpy as np
import torch
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore import context
from src.iouEval import iouEval_1
from src.util import getCityLossWeight, getBool, seed_seed
from src.model import ERFNet, Encoder_pred
from src.dataset import getCityScapesDataLoader_GeneratorDataset

# Pytorch NLLLoss + log_softmax
class SoftmaxCrossEntropyLoss(nn.Cell):

    def __init__(self, num_cls, weight):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.unsqueeze = ops.ExpandDims()
        self.get_size = ops.Size()
        self.exp = ops.Exp()
        self.pow = ops.Pow()

        self.weight = weight
        if isinstance(self.weight, tuple):
            self.use_focal = True
            self.gamma = self.weight[0]
            self.alpha = self.weight[1]
        else:
            self.use_focal = False

    def construct(self, pred, labels):
        labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1,))
        pred = self.transpose(pred, (0, 2, 3, 1))
        pred = self.reshape(pred, (-1, self.num_cls))
        one_hot_labels = self.one_hot(labels, self.num_cls, self.on_value, self.off_value)
        pred = self.cast(pred, mstype.float32)
        num = self.get_size(labels)

        if self.use_focal:
            loss = self.ce(pred, one_hot_labels)
            factor = self.pow(1 - self.exp(-loss), self.gamma) * self.alpha
            loss = self.div(self.sum(factor * loss), num)
            return loss

        if self.weight is not None:
            weight = mnp.copy(self.weight)
            weight = self.cast(weight, mstype.float32)
            weight = self.unsqueeze(weight, 0)
            expand = ops.BroadcastTo(pred.shape)
            weight = expand(weight)
            weight_masked = weight[mnp.arange(num), labels]
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss * weight_masked), self.sum(weight_masked))
        else:
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss), num)
        return loss

def IOU_1(network_trainefd, dataloader, num_class, enc):
    ioueval = iouEval_1(num_class)
    loss = SoftmaxCrossEntropyLoss(num_class, getCityLossWeight(enc))
    loss_list = []
    network_new = network_trainefd
    network_new.set_train(False)
    for index, (images, labels) in enumerate(dataloader):
        preds = network_new(images)
        l = loss(preds, labels)
        loss_list.append(float(str(l)))
        print("step {}/{}: loss:  ".format(index+1, dataloader.get_dataset_size()), l)
        preds = torch.Tensor(preds.asnumpy().argmax(axis=1).astype(np.int32)).unsqueeze(1).long()
        labels = torch.Tensor(labels.asnumpy().astype(np.int32)).unsqueeze(1).long()
        ioueval.addBatch(preds, labels)

    mean_iou, iou_class = ioueval.getIoU()
    mean_iou = mean_iou.item()
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_iou, mean_loss, iou_class

def evalNetwork(network, eval_dataloader, ckptPath, encode_1, num_class=20, weight_init="XavierUniform"):

    # load model checkpoint
    if ckptPath is None:
        print("no model checkpoint!")
    elif not os.path.exists(ckptPath):
        print("not exist {}".format(ckptPath))
    else:
        print("load model checkpoint {}!".format(ckptPath))
        param_dict = load_checkpoint(ckptPath)
        load_param_into_net(network, param_dict)

    mean_iou, mean_loss, iou_class = IOU_1(network, eval_dataloader, num_class, encode_1)
    os.path.splitext(ckptPath)
    with open(os.path.splitext(ckptPath)[0] + "_metric.txt", "w") as file:
        print("model path", ckptPath, file=file)
        print("mean_iou", mean_iou, file=file)
        print("mean_loss", mean_loss, file=file)
        print("iou_class", iou_class, file=file)

def listCKPTPath(model_root_path_1, enc):
    paths_1 = []
    names = os.listdir(model_root_path_1)
    for name in names:
        if name.endswith(".ckpt") and name+".metric.txt" not in names:
            if enc and name.startswith("Encoder"):
                ckpt_path = os.path.join(model_root_path_1, name)
                paths_1.append(ckpt_path)
            elif not enc and name.startswith("ERFNet"):
                ckpt_path = os.path.join(model_root_path_1, name)
                paths_1.append(ckpt_path)
    return paths_1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--run_distribute', type=str)
    parser.add_argument('--device_target', default="Ascend", type=str)
    parser.add_argument('--encode', type=str)
    parser.add_argument('--model_root_path', type=str)
    parser.add_argument('--device_id', type=int)

    config = parser.parse_args()
    model_root_path = config.model_root_path
    encode_ = getBool(config.encode)
    device_id = config.device_id
    CityScapesRoot = config.data_path
    run_distribute = getBool(config.run_distribute)

    seed_seed()
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target=config.device_target)
    context.set_context(save_graphs=False)

    eval_dataloader_1 = getCityScapesDataLoader_GeneratorDataset(CityScapesRoot, "val", 6, \
                                                               encode_, 512, False, False)

    weight_init_1 = "XavierUniform"
    if encode_:
        network_1 = Encoder_pred(stage=1, num_class=20, weight_init=weight_init_1, \
            run_distribute=False, train=False)
    else:
        network_1 = ERFNet(stage=1, num_class=20, init_conv=weight_init_1, run_distribute=False, \
            train=False)

    if not run_distribute:
        rank_id = 0
        rank_size = 1
        context.set_context(device_id=device_id)
        if os.path.isdir(model_root_path):
            paths = listCKPTPath(model_root_path, encode_)
            for path in paths:
                evalNetwork(network_1, eval_dataloader_1, path, encode_)
        else:
            evalNetwork(network_1, eval_dataloader_1, model_root_path, encode_)
    else:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        ckpt_files_path = listCKPTPath(model_root_path, encode_)
        n = math.ceil(len(ckpt_files_path) / rank_size)
        ckpt_files_path = ckpt_files_path[rank_id*n : rank_id*n + n]
        for path in ckpt_files_path:
            evalNetwork(network_1, eval_dataloader_1, path, encode_)
        