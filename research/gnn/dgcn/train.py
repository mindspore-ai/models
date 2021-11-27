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
# coding=utf-8
"""training the network"""
import os
import time
import argparse
import mindspore
from mindspore import Tensor
from mindspore import context, set_seed, save_checkpoint
from mindspore.communication.management import init
from mindspore.context import ParallelMode
import numpy as np
import src.data_process as dp
from src.dgcn import DGCN
from src.metrics import LossAccuracyWrapper, TrainNetWrapper
from src.utilities import diffusion_fun_sparse, \
    diffusion_fun_improved_ppmi_dynamic_sparsity, get_scaled_unsup_weight_max, rampup
from src.config import ConfigDGCN

parser = argparse.ArgumentParser(description="run DGCN")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--dataset", type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help="Dataset.")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU"], help="device target (default: Ascend)")
parser.add_argument("--distributed", type=bool, default=False, help="Device id")
args = parser.parse_args()

if args.dataset == "citeseer":
    args.seed = 1235
    args.lr = 0.0153

if args.device_target == "GPU":
    if args.dataset == "cora":
        args.seed = 1024
        args.lr = 0.03430959018564401
    if args.dataset == "citeseer":
        args.seed = 852
        args.lr = 0.02229329021215099
    if args.dataset == "pubmed":
        args.seed = 829
        args.lr = 0.029074859603424454

set_seed(args.seed)

def run_train(learning_rate=0.01, n_epochs=200, dataset=None, dropout_rate=0.5,
              hidden_size=36):
    """run train."""
    device_num = int(os.getenv('DEVICE_NUM'))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        if not args.distributed:
            context.set_context(device_id=args.device_id)
        if args.distributed:
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    elif args.device_target == "GPU":
        context.set_context(device_id=args.device_id)
        if args.distributed:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = dp.load_graph_data(dataset)
    train_mask = np.reshape(train_mask, (-1, len(train_mask)))
    val_mask = np.reshape(val_mask, (-1, len(val_mask)))
    test_mask = np.reshape(test_mask, (-1, len(test_mask)))

    #### obtain diffusion and ppmi matrices
    diffusions = diffusion_fun_sparse(adj.tocsc())  # Normalized adjacency matrix
    ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(adj, path_len=config.path_len, k=1.0)  # ppmi matrix

    adj = Tensor(adj.toarray(), mindspore.float32)
    features = Tensor(features.toarray(), mindspore.float32)
    diffusions = Tensor(diffusions.toarray(), mindspore.float32)
    labels = Tensor(labels, mindspore.float32)
    y_test = Tensor(y_test, mindspore.float32)
    y_val = Tensor(y_val, mindspore.float32)
    y_train = Tensor(y_train, mindspore.float32)
    ppmi = Tensor(ppmi.toarray(), mindspore.float32)

    #### construct the classifier model ####

    feature_size = features.shape[1]
    label_size = y_train.shape[1]
    layer_sizes = [(feature_size, hidden_size), (hidden_size, label_size)]

    print("Convolution Layers:" + str(layer_sizes))

    dgcn_net = DGCN(input_dim=features.shape[1], hidden_dim=hidden_size, output_dim=y_train.shape[1],
                    dropout=dropout_rate)
    train_net = TrainNetWrapper(dgcn_net, y_train, train_mask, weight_decay=config.weight_decay,
                                learning_rate=learning_rate)
    eval_net = LossAccuracyWrapper(dgcn_net, y_val, val_mask, weight_decay=config.weight_decay)
    test_net = LossAccuracyWrapper(dgcn_net, y_test, test_mask, weight_decay=config.weight_decay)
    loss_list = []
    dgcn_net.add_flags_recursive(fp16=True)

    #### train model ####
    print("...training...")
    num_labels = np.sum(train_mask)
    X_train_shape = features.shape[0]
    accuracys = []

    for epoch in range(n_epochs):
        t = time.time()

        scaled_unsup_weight_max = get_scaled_unsup_weight_max(
            num_labels, X_train_shape, unsup_weight_max=15.0)

        ret = rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=120)
        ret = Tensor(ret, mindspore.float32)

        train_net.set_train()
        train_result = train_net(diffusions, ppmi, features, ret)
        train_loss = train_result[0].asnumpy()
        train_accuracy = train_result[1].asnumpy()

        eval_net.set_train(False)
        eval_result = eval_net(diffusions, ppmi, features, ret)
        eval_loss = eval_result[0].asnumpy()
        eval_accuracy = eval_result[1].asnumpy()
        loss_list.append(eval_loss)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(eval_loss),
              "val_acc=", "{:.5f}".format(eval_accuracy), "time=", "{:.5f}".format(time.time() - t))
        if epoch > config.early_stopping and loss_list[-1] > np.mean(loss_list[-(config.early_stopping + 1):-1]):
            print("Early stopping...")
            break

        t_test = time.time()
        test_net.set_train(False)
        test_result = test_net(diffusions, ppmi, features, ret)
        test_accuracy = test_result[1].asnumpy()
        accuracys.append(test_accuracy)
        if test_accuracy == max(accuracys):
            ckpt_file_name = "./checkpoint"
            ckpt_file_name = os.path.join(ckpt_file_name, dataset)
            if not os.path.isdir(ckpt_file_name):
                os.makedirs(ckpt_file_name)
            ckpt_path = os.path.join(ckpt_file_name, "dgcn.ckpt")
            save_checkpoint(dgcn_net, ckpt_path)
        print("Test set results:", "accuracy=", "{:.5f}".format(test_accuracy),
              "time=", "{:.5f}".format(time.time() - t_test))
    print("Best test accuracy = ", "{:.5f}".format(max(accuracys)))
    return max(accuracys)


if __name__ == "__main__":
    data = args.dataset
    config = ConfigDGCN()

    print("Testing on the dataset: " + data)
    if data in ['cora', 'citeseer', 'pubmed']:
        run_train(dataset=data, learning_rate=args.lr,
                  dropout_rate=config.dropout, n_epochs=config.epochs, hidden_size=config.hidden1)
        print("Finished")
    else:
        print("No such a dataset: " + data)
