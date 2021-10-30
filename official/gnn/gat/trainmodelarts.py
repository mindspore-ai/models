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
"""Test train gat"""
import argparse
import os

import numpy as np
from mindspore import context
from mindspore import export, Tensor
from mindspore.communication.management import init
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from src.gat_sdk import GAT
from src.model_utils.config import config
from src.writer import writer_data
from src.dataset import load_and_process
from src.utils import LossAccuracyWrapper, TrainGAT


def gnn_train():
    """Train GAT model."""
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--data_url', type=str, default='', help='Dataset directory')
    parser.add_argument('--train_url', type=str, default='', help='The path model saved')
    parser.add_argument('--train_nodes_num', type=int, default=140, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=500, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    args_opt, _ = parser.parse_known_args()

    # 训练数据预处理
    if not os.path.exists(os.path.join(args_opt.train_url, "data_mr")):
        os.mkdir(os.path.join(args_opt.train_url, "data_mr"))
    writer_data(mindrecord_script=args_opt.dataset,
                mindrecord_file=os.path.join(args_opt.train_url, "data_mr"),
                mindrecord_partitions=1,
                mindrecord_header_size_by_bit=18,
                mindrecord_page_size_by_bit=20,
                graph_api_args=args_opt.data_url)

    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False)
    init()

    # train parameters
    hid_units = config.hid_units
    n_heads = config.n_heads
    early_stopping = config.early_stopping
    lr = config.lr
    l2_coeff = config.l2_coeff
    num_epochs = config.num_epochs
    feature, biases, y_train, train_mask, y_val, eval_mask, y_test, test_mask = \
        load_and_process(os.path.join(args_opt.train_url, "data_mr", args_opt.dataset), args_opt.train_nodes_num, \
                         args_opt.eval_nodes_num, args_opt.test_nodes_num)
    feature_size = feature.shape[2]
    num_nodes = feature.shape[1]
    num_class = y_train.shape[2]

    gat_net = GAT(feature_size,
                  num_class,
                  num_nodes,
                  hid_units,
                  n_heads,
                  attn_drop=config.attn_dropout,
                  ftr_drop=config.feature_dropout)
    gat_net.add_flags_recursive(fp16=True)

    feature = Tensor(feature)
    biases = Tensor(biases)

    eval_net = LossAccuracyWrapper(gat_net, num_class, y_val,
                                   eval_mask, l2_coeff)

    train_net = TrainGAT(gat_net, num_class, y_train,
                         train_mask, lr, l2_coeff)

    train_net.set_train(True)
    val_acc_max = 0.0
    val_loss_min = np.inf

    if not os.path.exists(os.path.join(args_opt.train_url, "model")):
        os.makedirs(os.path.join(args_opt.train_url, "model"))

    for _epoch in range(num_epochs):
        train_result = train_net(feature, biases)
        train_loss = train_result[0].asnumpy()
        train_acc = train_result[1].asnumpy()

        eval_result = eval_net(feature, biases)
        eval_loss = eval_result[0].asnumpy()
        eval_acc = eval_result[1].asnumpy()

        print("Epoch:{}, train loss={:.5f}, train acc={:.5f} | val loss={:.5f}, val acc={:.5f}".format(
            _epoch, train_loss, train_acc, eval_loss, eval_acc))
        if eval_acc >= val_acc_max or eval_loss < val_loss_min:
            if eval_acc >= val_acc_max and eval_loss < val_loss_min:
                val_acc_model = eval_acc
                val_loss_model = eval_loss
                if os.path.exists(os.path.join(args_opt.train_url, "model", (args_opt.dataset + '.ckpt'))):
                    os.remove(os.path.join(args_opt.train_url, "model", (args_opt.dataset + '.ckpt')))
                save_checkpoint(train_net.network,
                                os.path.join(args_opt.train_url, "model", (args_opt.dataset + '.ckpt')))
            val_acc_max = np.max((val_acc_max, eval_acc))
            val_loss_min = np.min((val_loss_min, eval_loss))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == early_stopping:
                print("Early Stop Triggered!, Min loss: {}, Max accuracy: {}".format(val_loss_min, val_acc_max))
                print("Early stop model validation loss: {}, accuracy{}".format(val_loss_model, val_acc_model))
                break
    gat_net_test = GAT(feature_size,
                       num_class,
                       num_nodes,
                       hid_units,
                       n_heads,
                       attn_drop=0.0,
                       ftr_drop=0.0)
    load_checkpoint(os.path.join(args_opt.train_url, "model", (args_opt.dataset + '.ckpt')), net=gat_net_test)
    gat_net_test.add_flags_recursive(fp16=True)

    test_net = LossAccuracyWrapper(gat_net_test,
                                   num_class,
                                   y_test,
                                   test_mask,
                                   l2_coeff)
    test_result = test_net(feature, biases)
    print("Test loss={}, test acc={}".format(test_result[0], test_result[1]))

    # # 模型冻结
    adj_tensor = Tensor(np.zeros((1, num_nodes * num_nodes), np.float32))
    feature_tensor = Tensor(np.zeros((1, num_nodes * feature_size), np.float32))
    export(gat_net_test, feature_tensor, adj_tensor,
           file_name=os.path.join(args_opt.train_url, "model", args_opt.dataset), file_format="AIR")
    export(gat_net_test, feature_tensor, adj_tensor,
           file_name=os.path.join(args_opt.train_url, "model", args_opt.dataset), file_format="MINDIR")


if __name__ == "__main__":
    gnn_train()
