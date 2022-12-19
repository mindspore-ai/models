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

"""Knowledge Distillation."""


import os
import argparse
import numpy as np

from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.data import load_data
from src.resnet import CifarResNet
from src.vgg import CifarVGG
from src.distill import Distill


def main_distill(args):
    print("args:", args)

    # data
    train_set, test_set = load_data(
        fdir=args.data_dir,
        dset=args.dataset,
        download=args.download,
        batch_size=args.batch_size
    )

    # load teacher model
    t_net_name = args.t_net_name
    if t_net_name == "ResNet14":
        tmodel = CifarResNet(
            res_n_layer=14,
            n_classes=args.n_classes,
        )
    elif t_net_name == "ResNet110":
        tmodel = CifarResNet(
            res_n_layer=110,
            n_classes=args.n_classes,
        )
    else:
        raise ValueError("No such t_net: ", t_net_name)

    n_params = sum([
        np.prod(param.shape) for param in tmodel.get_parameters()
    ])
    print("Total number of parameters : ", n_params)

    fpath = os.path.join(
        args.ckpt_dir,
        "{}-{}.ckpt".format(args.dataset, args.t_net_name)
    )
    param_dict = load_checkpoint(fpath)
    load_param_into_net(tmodel, param_dict)
    tmodel.set_train(False)

    # load model
    net_name = args.net_name
    if net_name == "VGG8":
        model = CifarVGG(
            vgg_n_layer=8,
            n_classes=args.n_classes,
        )
    else:
        raise ValueError("No such net: ", net_name)

    n_params = sum([
        np.prod(param.shape) for param in model.get_parameters()
    ])
    print("Total number of parameters : ", n_params)

    algo = Distill(
        train_set=train_set,
        test_set=test_set,
        model=model,
        tmodel=tmodel,
        dargs=args
    )
    algo.main()

    # save logs
    fpath = os.path.join(
        args.log_dir, args.log_name
    )
    algo.save_logs(fpath)

    # save ckpt
    fpath = os.path.join(
        args.ckpt_dir, args.ckpt_name
    )
    algo.save_ckpt(fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data path')
    parser.add_argument('--download', type=bool, default=True, help='download or not')
    parser.add_argument('--n_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--t_net', type=str, default='ResNet', help='teacher net type')
    parser.add_argument('--t_n_layer', type=int, default=14, help='number of teacher layers')
    parser.add_argument('--t_net_name', type=str, default='ResNet14', help='teacher net name')
    parser.add_argument('--net', type=str, default='VGG', help='student net type')
    parser.add_argument('--n_layer', type=int, default=8, help='number of student layers')
    parser.add_argument('--net_name', type=str, default='VGG8', help='student net name')
    parser.add_argument('--kd_way', type=str, default='ATS', help='TS or ATS')
    parser.add_argument('--s_tau', type=float, default=1.0, help='student temperature')
    parser.add_argument('--t_tau', type=float, default=3.0, help='teacher temperature for wrong classes')
    parser.add_argument('--tp_tau', type=float, default=4.0, help='teacher temperature for correct class')
    parser.add_argument('--lamb', type=float, default=0.5, help='balance factor of CE loss and KD loss')
    parser.add_argument('--epoches', type=int, default=240, help='number of train epoches')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of sgd')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='log path')
    parser.add_argument('--log_name', type=str, default='classify.log', help='log file name')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str, default='cifar100-ResNet14.ckpt', help='checkpoint fname')

    args0 = parser.parse_args()
    main_distill(args0)
