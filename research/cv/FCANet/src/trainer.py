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
"""
the trainer class for train.py and eval.py
"""
from copy import deepcopy
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.ops.operations import SigmoidCrossEntropyWithLogits
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.train.serialization import save_checkpoint, load_checkpoint

from src import helpers
from src import my_custom_transforms as mtr
from src.dataloader_cut import GeneralCutDataset
from src.model import fcanet

random.seed(10)

class Trainer():
    """ Trainer for training and eval"""
    def __init__(self, p):
        self.p = p
        context.set_context(mode=context.GRAPH_MODE, device_target=p["device"])

        # set train and eval data and dataloader
        transform_train = mtr.Compose(
            [
                mtr.MatchShortSideResize(size=p["size"][0]),
                mtr.ITIS_Crop(
                    itis_pro=p["itis_pro"], mode="strategy#05", crop_size=p["size"]
                ),
                mtr.CatPointMask(mode="DISTANCE_POINT_MASK_SRC"),
                mtr.CatPointMask(mode="DISTANCE_POINT_MASK_FIRST"),
                mtr.GeneLossWeight(),
                mtr.Transfer(),
                mtr.Decouple(),
            ]
        )

        self.train_set = GeneralCutDataset(
            p["dataset_path"],
            p["dataset_train"],
            "train.txt",
            transform=transform_train,
            max_num=p["max_num"],
            batch_size=-self.p["batch_size"],
        )
        self.train_loader = ds.GeneratorDataset(
            self.train_set,
            [
                "img",
                "gt",
                "pos_points_mask",
                "neg_points_mask",
                "pos_mask_dist_src",
                "neg_mask_dist_src",
                "pos_mask_dist_first",
                "click_loss_weight",
                "first_loss_weight",
                "id_num",
                "crop_lt",
                "flip",
            ],
            num_parallel_workers=p["num_workers"],
        )

        self.train_loader = self.train_loader.shuffle(buffer_size=5)
        self.train_loader = self.train_loader.batch(
            p["batch_size"], drop_remainder=True
        )
        self.train_loader = self.train_loader.repeat(1)

        self.val_robot_sets = []
        for dataset_val in p["datasets_val"]:
            self.val_robot_sets.append(
                GeneralCutDataset(
                    p["dataset_path"],
                    dataset_val,
                    "val.txt",
                    transform=None,
                    max_num=p["max_num"],
                    batch_size=0,
                )
            )

        # set network
        self.model = fcanet.FCANet(
            size=p["size"][0], backbone_pretrained=p["backbone_pretrained"]
        )

        # set loss function and learing rate scheduler
        self.criterion = SigmoidCrossEntropyWithLogits()
        self.scheduler = helpers.PolyLR(
            epoch_max=30, base_lr=p["lr"], power=0.9, cutoff_epoch=29
        )
        self.best_metric = [-1.0, -1.0]

        # resume from checkpoint
        if p["resume"] is not None:
            load_checkpoint(p["resume"], net=self.model)
            print("Load model from [{}]".format(p["resume"]))

    def training(self, epoch):
        """ train one epoch"""
        print("Training :")
        mtr.current_epoch = epoch
        loss_total = 0

        # set one training step
        backbone_params = list(
            filter(lambda x: "resnet" in x.name, self.model.trainable_params())
        )
        other_params = list(
            filter(lambda x: "resnet" not in x.name, self.model.trainable_params())
        )

        group_params = [
            {"params": backbone_params, "lr": self.scheduler.get_lr()},
            {"params": other_params, "lr": 1 * self.scheduler.get_lr()},
            {"order_params": self.model.trainable_params()},
        ]

        optimizer = nn.SGD(
            group_params,
            learning_rate=self.scheduler.get_lr(),
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        )

        tmptmp = fcanet.MyWithLossCell(
            self.model, self.criterion, self.p["batch_size"], self.p["size"][0]
        )
        trainonestep = fcanet.MyTrainOneStepCell(
            tmptmp, self.model, self.criterion, optimizer
        )
        trainonestep.set_train(True)
        # train process
        tbar = tqdm(total=self.train_loader.get_dataset_size())

        i = 0

        for i, sample_batched in enumerate(self.train_loader.create_dict_iterator()):
            (
                img,
                gt,
                pos_mask_dist_src,
                neg_mask_dist_src,
                pos_mask_dist_first,
                click_loss_weight,
                first_loss_weight,
            ) = [
                sample_batched[k]
                for k in [
                    "img",
                    "gt",
                    "pos_mask_dist_src",
                    "neg_mask_dist_src",
                    "pos_mask_dist_first",
                    "click_loss_weight",
                    "first_loss_weight",
                ]
            ]

            loss = trainonestep(
                img,
                pos_mask_dist_src,
                neg_mask_dist_src,
                pos_mask_dist_first,
                gt,
                click_loss_weight,
                first_loss_weight,
            )

            tbar.update(1)
            loss_total += loss.asnumpy()
            tbar.set_description("Loss: %.3f" % (loss_total / (i + 1)))

            output = tmptmp.out_tmp
            output = P.Sigmoid()(output)

            # record the previous result for iterative training
            if self.p["itis_pro"] > 0:
                current_batchsize = sample_batched["pos_points_mask"].shape[0]
                for j in range(current_batchsize):
                    gt = (sample_batched["gt"].asnumpy()[j, 0, :, :] > 0.5).astype(
                        np.uint8
                    )
                    pos_points_mask = (
                        sample_batched["pos_points_mask"].asnumpy()[j, 0, :, :] > 0.0001
                    ).astype(np.uint8)
                    neg_points_mask = (
                        sample_batched["neg_points_mask"].asnumpy()[j, 0, :, :] > 0.0001
                    ).astype(np.uint8)

                    result = output.asnumpy()[j, 0, :, :]

                    pred = (result > 0.2).astype(np.uint8)

                    pt, if_pos = helpers.get_anno_point(
                        pred, gt, np.maximum(pos_points_mask, neg_points_mask)
                    )
                    if if_pos:
                        pos_points_mask[pt[1], pt[0]] = 1
                    else:
                        neg_points_mask[pt[1], pt[0]] = 1

                    object_id = self.train_set.ids_list[sample_batched["id_num"].asnumpy()[j]]

                    mtr.record_anno[object_id] = [
                        helpers.get_points_list(pos_points_mask),
                        helpers.get_points_list(neg_points_mask),
                    ]
                    mtr.record_crop_lt[object_id] = list(
                        sample_batched["crop_lt"].asnumpy()[j]
                    )
                    mtr.record_if_flip[object_id] = int(sample_batched["flip"].asnumpy()[j])

        tbar.close()
        print("Loss: %.3f" % (loss_total / (i + 1)))

    def validation_robot(self, epoch, tsh=0.5, resize=None):
        """ validation with robot user"""
        self.model.set_train(False)
        print("+" * 79)
        for index, val_robot_set in enumerate(self.val_robot_sets):
            print("Validation Robot: [{}] ".format(self.p["datasets_val"][index]))
            self.validation_robot_dataset(index, val_robot_set, tsh, resize)

    def validation_robot_dataset(self, index, val_robot_set, tsh, resize):
        """ validation with robot user for each dataset """
        tbar = tqdm(val_robot_set)
        img_num = len(val_robot_set)
        point_num_target_sum, pos_points_num, neg_points_num = 0, 0, 0
        point_num_miou_sum = [0] * (self.p["record_point_num"] + 1)
        for _, sample in enumerate(tbar):
            gt = np.array(Image.open(sample["meta"]["gt_path"]))
            pred = np.zeros_like(gt)
            pos_points, neg_points = [], []
            if_get_target = False
            for point_num in range(1, self.p["max_point_num"] + 1):
                pt, if_pos = helpers.get_anno_point(
                    pred, gt, pos_points + neg_points
                )
                if if_pos:
                    pos_points.append(pt)
                    if not if_get_target:
                        pos_points_num += 1
                else:
                    neg_points.append(pt)
                    if not if_get_target:
                        neg_points_num += 1

                sample_cpy = deepcopy(sample)
                sample_cpy["pos_points_mask"] = helpers.get_points_mask(
                    gt.shape[::-1], pos_points
                )
                sample_cpy["neg_points_mask"] = helpers.get_points_mask(
                    gt.shape[::-1], neg_points
                )

                if resize is not None:
                    if isinstance(resize, int):
                        short_len = min(gt.shape[0], gt.shape[1])
                        dsize = (
                            int(gt.shape[1] * resize / short_len),
                            int(gt.shape[0] * resize / short_len),
                        )
                    elif isinstance(resize, tuple):
                        dsize = resize

                    mtr.Resize(dsize)(sample_cpy)
                mtr.CatPointMask(mode="DISTANCE_POINT_MASK_SRC", if_repair=False)(
                    sample_cpy
                )

                if point_num == 1:
                    pos_mask_first = sample_cpy["pos_points_mask"].copy()

                sample_cpy["pos_mask_dist_first"] = (
                    np.minimum(distance_transform_edt(1 - pos_mask_first), 255.0)
                    * 255.0
                )

                mtr.Transfer()(sample_cpy)

                img = Tensor(sample_cpy["img"][None, :, :, :], mstype.float32)
                pos_mask_dist_src = Tensor(
                    sample_cpy["pos_mask_dist_src"][None, :, :, :], mstype.float32
                )
                neg_mask_dist_src = Tensor(
                    sample_cpy["neg_mask_dist_src"][None, :, :, :], mstype.float32
                )
                pos_mask_dist_first = Tensor(
                    sample_cpy["pos_mask_dist_first"][None, :, :, :], mstype.float32
                )

                output = self.model(
                    img, pos_mask_dist_src, neg_mask_dist_src, pos_mask_dist_first
                )

                output = output[0]
                output = P.Sigmoid()(output)

                result = output.asnumpy()[0, 0, :, :]

                if resize is not None:
                    result = cv2.resize(
                        result, gt.shape[::-1], interpolation=cv2.INTER_LINEAR
                    )

                pred = (result > tsh).astype(np.uint8)
                miou = ((pred == 1) & (gt == 1)).sum() / (
                    ((pred == 1) | (gt == 1)) & (gt != 255)
                ).sum()

                if point_num <= self.p["record_point_num"]:
                    point_num_miou_sum[point_num] += miou

                if (not if_get_target) and (
                        miou >= self.p["miou_target"][index]
                        or point_num == self.p["max_point_num"]
                ):
                    point_num_target_sum += point_num
                    if_get_target = True

                if if_get_target and point_num >= self.p["record_point_num"]:
                    break

        print("(point_num_target_avg : {})".format(point_num_target_sum / img_num))
        print(
            "(pos_points_num_avg : {}) (neg_points_num_avg : {})".format(
                pos_points_num / img_num, neg_points_num / img_num
            )
        )
        print(
            "(point_num_miou_avg : {})\n".format(
                np.array([round(i / img_num, 3) for i in point_num_miou_sum])
            )
        )

    def save_model_ckpt(self, path):
        """ save checkpoint"""
        save_checkpoint(self.model, path)
