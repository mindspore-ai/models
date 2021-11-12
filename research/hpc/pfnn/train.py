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
"""Run PFNN"""
import time
import argparse
import os
import numpy as np
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore import load_param_into_net, load_checkpoint
from mindspore.train.model import Model
from src import pfnnmodel, callback
from data import gendata, dataset


def calerror(netg_calerror, netf_calerror, lenfac_calerror, TeSet_calerror):
    """
    The eval function
    Args:
        netg: Instantiation of NetG
        netf: Instantiation of NetF
        lenfac: Instantiation of LenFac
        TeSet: Test Dataset
    Return:
        error: Test error
    """
    x = Tensor(TeSet_calerror.x, mstype.float32)
    TeSet_u = (netg_calerror(x) + lenfac_calerror(Tensor(x[:, 0])).reshape(-1, 1) * netf_calerror(x)).asnumpy()
    Teerror = (((TeSet_u - TeSet_calerror.ua)**2).sum() /
               (TeSet_calerror.ua).sum()) ** 0.5
    return Teerror


def train_netg(args_netg, net_netg, optim_netg, dataset_netg):
    """
    The process of preprocess and process to train NetG

    Args:
        net(NetG): The instantiation of NetG
        optim: The optimizer to optimal NetG
        dataset: The traing dataset of NetG
    """
    print("START TRAIN NEURAL NETWORK G")
    model = Model(network=net_netg, loss_fn=None, optimizer=optim_netg)
    model.train(args_netg.g_epochs, dataset_netg, callbacks=[
                callback.SaveCallbackNETG(net_netg, args_netg.g_path)])


def train_netloss(args_netloss, netg_netloss, netf_netloss, netloss_netloss,
                  lenfac_netloss, optim_netloss, InSet_netloss, BdSet_netloss, dataset_netloss):
    """
    The process of preprocess and process to train NetF/NetLoss

    Args:
        netg: The Instantiation of NetG
        netf: The Instantiation of NetF
        netloss: The Instantiation of NetLoss
        lenfac: The Instantiation of lenfac
        optim: The optimizer to optimal NetF/NetLoss
        dataset: The trainging dataset of NetF/NetLoss
    """
    grad_ = ops.composite.GradOperation(get_all=True)

    InSet_l = lenfac_netloss(Tensor(InSet.x[:, 0], mstype.float32))
    InSet_l = InSet_l.reshape((len(InSet_l), 1))
    InSet_lx = grad_(lenfac_netloss)(Tensor(InSet.x[:, 0], mstype.float32))[
        0].asnumpy()[:, np.newaxis]
    InSet_lx = np.hstack((InSet_lx, np.zeros(InSet_lx.shape)))
    BdSet_nl = lenfac_netloss(Tensor(BdSet_netloss.n_x[:, 0], mstype.float32)).asnumpy()[
        :, np.newaxis]

    load_param_into_net(netg_netloss, load_checkpoint(
        args_netloss.g_path), strict_load=True)
    InSet_g = netg_netloss(Tensor(InSet.x, mstype.float32))
    InSet_gx = grad_(netg_netloss)(Tensor(InSet.x, mstype.float32))[0]
    BdSet_ng = netg_netloss(Tensor(BdSet_netloss.n_x, mstype.float32))

    netloss_netloss.get_variable(InSet_g, InSet_l, InSet_gx, InSet_lx, InSet_netloss.a,
                                 InSet_netloss.size, InSet_netloss.dim, InSet_netloss.area, InSet_netloss.c,
                                 BdSet_netloss.n_length, BdSet_netloss.n_r, BdSet_nl, BdSet_ng)
    print("START TRAIN NEURAL NETWORK F")
    model = Model(network=netloss_netloss,
                  loss_fn=None, optimizer=optim_netloss)
    model.train(args_netloss.f_epochs, dataset_netloss, callbacks=[callback.SaveCallbackNETLoss(
        netf_netloss, args_netloss.f_path, InSet_netloss.x, InSet_l, InSet_g, InSet_netloss.ua)])


def trainer(args_er, netg_er, netf_er, netloss_er, lenfac_er, optim_g_er, optim_f_er,
            InSet_er, BdSet_er, dataset_g_er, dataset_loss_er):
    """
    The traing process that's includes traning network G and network F/Loss
    """
    print("START TRAINING")
    start_gnet_time = time.time()
    train_netg(args_er, netg_er, optim_g_er, dataset_g_er)
    elapsed_gnet = time.time() - start_gnet_time

    start_fnet_time = time.time()
    train_netloss(args_er, netg_er, netf_er, netloss_er, lenfac_er, optim_f_er,
                  InSet_er, BdSet_er, dataset_loss_er)
    elapsed_fnet = time.time() - start_fnet_time
    return elapsed_gnet, elapsed_fnet


def ArgParse():
    """Get Args"""
    parser = argparse.ArgumentParser(
        description="Penalty-Free Neural Network Method")
    parser.add_argument("--problem", type=int, default=1,
                        help="the type of the problem")
    parser.add_argument("--bound", type=float, default=[-1.0, 1.0, -1.0, 1.0],
                        help="lower and upper bound of the domain")
    parser.add_argument("--inset_nx", type=int, default=[60, 60],
                        help="size of the inner set")
    parser.add_argument("--bdset_nx", type=int, default=[60, 60],
                        help="size of the boundary set")
    parser.add_argument("--teset_nx", type=int, default=[101, 101],
                        help="size of the test set")
    parser.add_argument("--g_epochs", type=int, default=6000,
                        help="number of epochs to train neural network g")
    parser.add_argument("--f_epochs", type=int, default=6000,
                        help="number of epochs to train neural network f")
    parser.add_argument("--g_lr", type=float, default=0.01,
                        help="learning rate to train neural network g")
    parser.add_argument("--f_lr", type=float, default=0.01,
                        help="learning rate to train neural network f")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"],
                        help="use cpu or gpu to train")
    parser.add_argument("--tests_num", type=int, default=5,
                        help="number of independent tests")
    parser.add_argument("--path", type=str, default="./optimal_state/",
                        help="the basic folder of g_path and f_path")
    parser.add_argument("--g_path", type=str, default="optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netg")
    parser.add_argument("--f_path", type=str, default="optimal_state_f_pfnn.ckpt",
                        help="the path that will put checkpoint of netf")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = ArgParse()
    if args.device == "gpu":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    errors = np.zeros(args.tests_num)

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    args.g_path = args.path + args.g_path
    args.f_path = args.path + args.f_path

    for ii in range(args.tests_num):
        InSet, BdSet, TeSet = gendata.GenerateSet(args)
        dsg, dsloss = dataset.GenerateDataSet(InSet, BdSet)

        lenfac = pfnnmodel.LenFac(
            Tensor(args.bound, mstype.float32).reshape(2, 2), 1)
        netg = pfnnmodel.NetG()
        netf = pfnnmodel.NetF()
        netloss = pfnnmodel.Loss(netf)
        optimg = nn.Adam(netg.trainable_params(), learning_rate=args.g_lr)
        optimf = nn.Adam(netf.trainable_params(), learning_rate=args.f_lr)

        NetGTime, NetFTime = trainer(
            args, netg, netf, netloss, lenfac, optimg, optimf, InSet, BdSet, dsg, dsloss)
        print("Train NetG total time: %.2f, train NetG one step time: %.5f" %
              (NetGTime, NetGTime/args.g_epochs))
        print("Train NetF total time: %.2f, train NetF one step time: %.5f" %
              (NetFTime, NetFTime/args.f_epochs))

        load_param_into_net(netg, load_checkpoint(
            args.g_path), strict_load=True)
        load_param_into_net(netf, load_checkpoint(
            args.f_path), strict_load=True)
        errors[ii] = calerror(netg, netf, lenfac, TeSet)
        print("test_error = %.3e\n" % (errors[ii].item()))

    print(errors)
    errors_mean = errors.mean()
    errors_std = errors.std()

    print("test_error_mean = %.3e, test_error_std = %.3e"
          % (errors_mean.item(), errors_std.item()))
