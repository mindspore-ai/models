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
GCN training script.
"""
import os
import time
import argparse
import ast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn import manifold
from mindspore import context
from mindspore import Tensor, export
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from mindspore.communication.management import init
from src.writer import writer_data
from src.gcn_modelarts import GCN
from src.metrics import LossAccuracyWrapper, TrainNetWrapper
from src.config import ConfigGCN
from src.dataset import get_adj_features_labels, get_mask


def t_SNE(out_feature, dim):
    t_sne = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    return t_sne.fit_transform(out_feature)


def update_graph(i, data, scat, plot):
    scat.set_offsets(data[i])
    plt.title('t-SNE visualization of Epoch:{0}'.format(i))
    return scat, plot


def w2txt(file, data):
    s = ""
    for i in range(len(data)):
        s = s + str(data[i])
        s = s + " "
    with open(file, "w") as f:
        f.write(s)


def train():
    """Train model."""
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='obs://hw2czq/path/gcn/data_mr/', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='obs://hw2czq/path/gcn/output', help='The path model saved')
    parser.add_argument('--train_nodes_num', type=int, default=140, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=500, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    parser.add_argument('--save_TSNE', type=ast.literal_eval, default=False, help='Whether to save t-SNE graph')
    args_opt, _ = parser.parse_known_args()

    # 训练数据预处理
    if not os.path.exists(os.path.join(args_opt.output_dir, "data_mr")):
        os.mkdir(os.path.join(args_opt.output_dir, "data_mr"))
    writer_data(mindrecord_script=args_opt.dataset,
                mindrecord_file=os.path.join(args_opt.output_dir, "data_mr"),
                mindrecord_partitions=1,
                mindrecord_header_size_by_bit=18,
                mindrecord_page_size_by_bit=20,
                graph_api_args=args_opt.data_dir)

    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", save_graphs=False)
    init()
    config = ConfigGCN()
    adj, feature, label_onehot, label = get_adj_features_labels(
        os.path.join(args_opt.output_dir, "data_mr", args_opt.dataset))

    nodes_num = label_onehot.shape[0]
    train_mask = get_mask(nodes_num, 0, args_opt.train_nodes_num)
    eval_mask = get_mask(nodes_num, args_opt.train_nodes_num, args_opt.train_nodes_num + args_opt.eval_nodes_num)
    test_mask = get_mask(nodes_num, nodes_num - args_opt.test_nodes_num, nodes_num)

    class_num = label_onehot.shape[1]
    node_nums = feature.shape[0]
    input_dim = feature.shape[1]
    gcn_net = GCN(config, input_dim, class_num, node_nums)
    gcn_net.add_flags_recursive(fp16=True)

    adj = Tensor(adj)
    feature = Tensor(feature)

    eval_net = LossAccuracyWrapper(gcn_net, label_onehot, eval_mask, config.weight_decay)
    train_net = TrainNetWrapper(gcn_net, label_onehot, train_mask, config)

    loss_list = []

    if args_opt.save_TSNE:
        out_feature = gcn_net()
        tsne_result = t_SNE(out_feature.asnumpy(), 2)
        graph_data = []
        graph_data.append(tsne_result)
        fig = plt.figure()
        scat = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=2, c=label, cmap='rainbow')
        plt.title('t-SNE visualization of Epoch:0', fontsize='large', fontweight='bold', verticalalignment='center')

    for epoch in range(config.epochs):
        t = time.time()

        train_net.set_train()
        train_result = train_net(adj, feature)
        train_loss = train_result[0].asnumpy()
        train_accuracy = train_result[1].asnumpy()

        eval_net.set_train(False)
        eval_result = eval_net(adj, feature)
        eval_loss = eval_result[0].asnumpy()
        eval_accuracy = eval_result[1].asnumpy()

        loss_list.append(eval_loss)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(eval_loss),
              "val_acc=", "{:.5f}".format(eval_accuracy), "time=", "{:.5f}".format(time.time() - t))

        if args_opt.save_TSNE:
            out_feature = gcn_net()
            tsne_result = t_SNE(out_feature.asnumpy(), 2)
            graph_data.append(tsne_result)

        if epoch > config.early_stopping and loss_list[-1] > np.mean(loss_list[-(config.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    # 保存模型
    if not os.path.exists(os.path.join(args_opt.output_dir, "model")):
        os.makedirs(os.path.join(args_opt.output_dir, "model"))
    save_checkpoint(gcn_net, os.path.join(args_opt.output_dir, "model", (args_opt.dataset + '.ckpt')))
    gcn_net_test = GCN(config, input_dim, class_num, node_nums)
    load_checkpoint(os.path.join(args_opt.output_dir, "model", (args_opt.dataset + '.ckpt')), net=gcn_net_test)
    gcn_net_test.add_flags_recursive(fp16=True)

    # 模型冻结
    adj_tensor = Tensor(np.zeros((1, node_nums * node_nums), np.float32))
    feature_tensor = Tensor(np.zeros((1, node_nums * input_dim), np.float32))
    gcn_net_test.set_train(False)
    load_checkpoint(os.path.join(args_opt.output_dir, "model", (args_opt.dataset + '.ckpt')), net=gcn_net_test)
    export(gcn_net_test, adj_tensor, feature_tensor,
           file_name=os.path.join(args_opt.output_dir, "model", args_opt.dataset), file_format="AIR")
    export(gcn_net_test, adj_tensor, feature_tensor,
           file_name=os.path.join(args_opt.output_dir, "model", args_opt.dataset), file_format="ONNX")
    export(gcn_net_test, adj_tensor, feature_tensor,
           file_name=os.path.join(args_opt.output_dir, "model", args_opt.dataset), file_format="MINDIR")

    # 精度测试
    test_net = LossAccuracyWrapper(gcn_net_test, label_onehot, test_mask, config.weight_decay)
    t_test = time.time()
    test_net.set_train(False)
    test_result = test_net(adj, feature)
    test_loss = test_result[0].asnumpy()
    test_accuracy = test_result[1].asnumpy()
    print("Test set results:", "loss=", "{:.5f}".format(test_loss),
          "accuracy=", "{:.5f}".format(test_accuracy), "time=", "{:.5f}".format(time.time() - t_test))

    if args_opt.save_TSNE:
        ani = animation.FuncAnimation(fig, update_graph, frames=range(config.epochs + 1), fargs=(graph_data, scat, plt))
        ani.save('t-SNE_visualization.gif', writer='imagemagick')


if __name__ == '__main__':
    train()
