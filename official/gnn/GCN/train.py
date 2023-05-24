# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn import manifold
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore import Model, context

from src.eval_callback import EvalCallBack
from src.gcn import GCN, GCN_GPU
from src.metrics import LossAccuracyWrapper, TrainNetWrapper, Loss_Gpu
from src.config import ConfigGCN
from src.dataset import get_adj_features_labels, get_mask

from model_utils.config import config as default_args
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


def t_SNE(out_feature, dim):
    t_sne = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    return t_sne.fit_transform(out_feature)


def update_graph(i, data, scat, plot):
    scat.set_offsets(data[i])
    plt.title('t-SNE visualization of Epoch:{0}'.format(i))
    return scat, plot


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_args.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if default_args.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_args.data_path, default_args.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_args.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if default_args.device_target == "Ascend" or default_args.device_target == "GPU":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    default_args.save_ckptpath = os.path.join(default_args.output_path, default_args.save_ckptpath)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_gpu_train():
    """Train model."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=default_args.device_target, save_graphs=False)
    config = ConfigGCN()
    if not os.path.exists(config.ckpt_dir):
        os.mkdir(config.ckpt_dir)
    adj, feature, label_onehot, _ = get_adj_features_labels(default_args.data_dir)
    feature_d = np.expand_dims(feature, axis=0)
    label_onehot_d = np.expand_dims(label_onehot, axis=0)
    data = {"feature": feature_d, "label": label_onehot_d}
    dataset = ds.NumpySlicesDataset(data=data)
    nodes_num = label_onehot.shape[0]
    eval_mask = get_mask(nodes_num, default_args.train_nodes_num,
                         default_args.train_nodes_num + default_args.eval_nodes_num)
    class_num = label_onehot.shape[1]
    input_dim = feature.shape[1]
    adj = Tensor(adj, dtype=mstype.float32)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_steps,
                                   keep_checkpoint_max=config.keep_ckpt_max)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_gcn',
                                 directory=config.ckpt_dir,
                                 config=ckpt_config)
    gcn_net = GCN_GPU(config, input_dim, class_num, adj)
    cb = [TimeMonitor(), LossMonitor(), ckpoint_cb]
    opt = nn.Adam(gcn_net.trainable_params(), learning_rate=config.learning_rate)
    criterion = Loss_Gpu(eval_mask, config.weight_decay, gcn_net.trainable_params()[0])
    model = Model(gcn_net, loss_fn=criterion, optimizer=opt, amp_level="O3")
    if default_args.train_with_eval:
        metrics_name = "Acc"
        eval_metrics = {metrics_name: nn.Accuracy()}
        eval_model = Model(gcn_net, loss_fn=criterion, metrics=eval_metrics)
        eval_param_dict = {"model": eval_model, "dataset": ds.NumpySlicesDataset(data=data),
                           "metrics_name": metrics_name}

        def apply_eval(eval_param):
            apply_eval_model = eval_param["model"]
            apply_eval_dataset = eval_param["dataset"]
            res = apply_eval_model.eval(apply_eval_dataset, dataset_sink_mode=True)
            return res[eval_param["metrics_name"]]

        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=default_args.eval_start_epoch, save_best_ckpt=config.save_best_ckpt,
                               ckpt_directory=config.best_ckpt_dir, best_ckpt_name=config.best_ckpt_name,
                               metrics_name=metrics_name)
        cb.append(eval_cb)
    model.train(config.epochs, dataset, callbacks=cb, dataset_sink_mode=True)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """Train model."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=default_args.device_target, save_graphs=False)
    config = ConfigGCN()
    adj, feature, label_onehot, label = get_adj_features_labels(default_args.data_dir)

    nodes_num = label_onehot.shape[0]
    train_mask = get_mask(nodes_num, 0, default_args.train_nodes_num)
    eval_mask = get_mask(nodes_num, default_args.train_nodes_num,
                         default_args.train_nodes_num + default_args.eval_nodes_num)
    test_mask = get_mask(nodes_num, nodes_num - default_args.test_nodes_num, nodes_num)

    class_num = label_onehot.shape[1]
    input_dim = feature.shape[1]
    gcn_net = GCN(config, input_dim, class_num)
    gcn_net.add_flags_recursive(fp16=True)

    adj = Tensor(adj)
    feature = Tensor(feature)

    eval_net = LossAccuracyWrapper(gcn_net, label_onehot, eval_mask, config.weight_decay)
    train_net = TrainNetWrapper(gcn_net, label_onehot, train_mask, config)

    loss_list = []

    if default_args.save_TSNE:
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

        if default_args.save_TSNE:
            out_feature = gcn_net()
            tsne_result = t_SNE(out_feature.asnumpy(), 2)
            graph_data.append(tsne_result)

        if epoch > config.early_stopping and loss_list[-1] > np.mean(loss_list[-(config.early_stopping+1):-1]):
            print("Early stopping...")
            break
    if not os.path.isdir(default_args.save_ckptpath):
        os.makedirs(default_args.save_ckptpath)
    ckpt_path = os.path.join(default_args.save_ckptpath, "gcn.ckpt")
    save_checkpoint(gcn_net, ckpt_path)
    gcn_net_test = GCN(config, input_dim, class_num)
    load_checkpoint(ckpt_path, net=gcn_net_test)
    gcn_net_test.add_flags_recursive(fp16=True)

    test_net = LossAccuracyWrapper(gcn_net_test, label_onehot, test_mask, config.weight_decay)
    t_test = time.time()
    test_net.set_train(False)
    test_result = test_net(adj, feature)
    test_loss = test_result[0].asnumpy()
    test_accuracy = test_result[1].asnumpy()
    print("Test set results:", "loss=", "{:.5f}".format(test_loss),
          "accuracy=", "{:.5f}".format(test_accuracy), "time=", "{:.5f}".format(time.time() - t_test))

    if default_args.save_TSNE:
        ani = animation.FuncAnimation(fig, update_graph, frames=range(config.epochs + 1), fargs=(graph_data, scat, plt))
        ani.save('t-SNE_visualization.gif', writer='imagemagick')


if __name__ == '__main__':
    if default_args.device_target == "GPU":
        run_gpu_train()
    else:
        run_train()
