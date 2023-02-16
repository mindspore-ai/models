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
""" selection samples to be make noise label in LR-EGAN model """
import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.train.callback import LossMonitor, TimeMonitor, Callback
from mindspore import Model
from mindspore import  load_checkpoint, load_param_into_net
from mindspore import context
from configs import parse_args
from preprocess import preprocess, parameter

from src.pyod_utils import standardizer
import pandas as pd

import seaborn as sns


class EvalCallback(Callback):
    """
    Evaluation per epoch, and save the best mse checkpoint.
    """

    def __init__(self, args, model, eval_ds, save_path="./"):

        self.model = model
        self.eval_ds = eval_ds
        self.best_mse = 999
        self.save_path = save_path
        self.args = args

    def epoch_end(self, run_context):
        """
        evaluate at epoch end.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        res = self.model.eval(self.eval_ds)
        mse = res["mse"]
        if mse < self.best_mse:
            self.best_mse = mse
            ms.save_checkpoint(cb_params.train_network, os.path.join(
                self.save_path, f"{self.args.data_name}_best_mse.ckpt"))
            print("the best epoch is", cur_epoch, "best mse is", self.best_mse)


class DatasetGenerator:
    ''' DatasetGenerator '''
    def __init__(self, data_x, data_y):

        self.data = data_x.astype(np.float32)
        self.label = data_y.astype(np.int32)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class AutoEncoder(nn.Cell):
    ''' AutoEncoder '''
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.Dense1 = nn.Dense(input_dim, input_dim//2)
        self.Dense2 = nn.Dense(input_dim//2, input_dim//4)
        self.Dense3 = nn.Dense(input_dim//4, input_dim//2)
        self.Dense4 = nn.Dense(input_dim//2, input_dim)
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

    def construct(self, x):
        x = self.Tanh(self.Dense1(x))
        x = self.Tanh(self.Dense2(x))
        x = self.Tanh(self.Dense3(x))
        x = self.Dense4(x)
        return x


def create_dataset(data, label, params, is_batch=True):
    dataset_generator = MyDataset(data, label)
    dataset = ds.GeneratorDataset(
        dataset_generator, ["data", "label"], shuffle=False)
    if is_batch:
        dataset = dataset.batch(params.batch_size)
    else:
        # take the whole data as one batch
        dataset = dataset.batch(data.shape[0])
    return dataset


def select_random(y_train, rate):
    # rate = a_adjust / a_original
    rate = 1 - rate
    index_abnormal = np.nonzero(y_train)
    index_abnormal = index_abnormal[0]
    np.random.seed(42)
    index = np.random.choice(index_abnormal.size, int(
        np.floor(rate * index_abnormal.size)), replace=False)
    y_train[index_abnormal[index]] = 0
    return y_train


def train_autoencoder(args, ds_train, ds_val, input_dim):

    autoencoder = AutoEncoder(input_dim)
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(autoencoder.trainable_params(), learning_rate=0.01)
    model = Model(autoencoder, loss_fn=loss_fn,
                  optimizer=optimizer, metrics={'mse'})
    eval_callback = EvalCallback(
        args, model, ds_val, save_path="./autoEncoder")
    model.train(epoch=100, train_dataset=ds_train, callbacks=[TimeMonitor(
        30), LossMonitor(100), eval_callback], dataset_sink_mode=False)


def select_autoencoder(X_train, y_train, rate, model_path, input_dim):
    ''' select samples with autoencoder '''
    rate = (1 - rate) * 100
    autoencoder = AutoEncoder(input_dim)
    autoencoder.set_train(False)
    param_dict = load_checkpoint(model_path)
    load_param_into_net(autoencoder, param_dict)
    index_abnormal = np.nonzero(y_train)[0]
    X_train_abnormal = X_train[index_abnormal]
    X_pred_abnormal = autoencoder(
        Tensor(X_train[index_abnormal], ms.float32)).asnumpy()
    mse = np.mean(np.power(X_train_abnormal - X_pred_abnormal, 2), axis=1)
    threshold = np.percentile(mse, rate)
    index = (mse <= threshold) * 1
    index = np.nonzero(index)[0]
    y_train[index_abnormal[index]] = 0
    return y_train


def draw_autoencoder(X_train, y_train, rate, model_path, figure_path):
    ''' draw picture using autoencoder '''
    rate = (1 - rate) * 100
    autoencoder = AutoEncoder(input_dim)
    autoencoder.set_train(False)
    param_dict = load_checkpoint(model_path)
    load_param_into_net(autoencoder, param_dict)
    index_abnormal = np.nonzero(y_train)
    index_abnormal = index_abnormal[0]
    X_train_abnormal = X_train[index_abnormal]
    X_pred_abnormal = autoencoder(
        Tensor(X_train[index_abnormal], ms.float32)).asnumpy()
    mse = np.mean(np.power(X_train_abnormal - X_pred_abnormal, 2), axis=1)
    threshold = np.percentile(mse, rate)
    plt.figure()
    sns.distplot(np.log(mse))
    plt.vlines(np.log(threshold), 0, 0.9, 'r')
    plt.savefig(figure_path)


if __name__ == "__main__":

    args_ = parse_args()

    if args_.mindspore_mode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE)
    else:
        context.set_context(mode=context.PYNATIVE_MODE)

    context.set_context(device_target=args_.device, device_id=args_.device_id)
    dataset_args = parameter()

    if args_.data_path[-1] != '/':
        args_.data_path = args_.data_path+'/'
    dataset_args.data_name = args_.data_name
    dataset_args.data_path = args_.data_path

    X_train_, y_train_, X_val, y_val, X_test, y_test = preprocess(dataset_args)

    X_train_norm, X_test_norm = standardizer(X_train_, X_test)
    X_train_norm_not_used, X_val_norm = standardizer(X_train_, X_val)
    X_train_pandas = pd.DataFrame(X_train_norm)
    X_test_pandas = pd.DataFrame(X_test_norm)
    X_val_pandas = pd.DataFrame(X_val_norm)
    X_train_pandas.fillna(X_train_pandas.mean(), inplace=True)
    X_test_pandas.fillna(X_train_pandas.mean(), inplace=True)
    X_val_pandas.fillna(X_val_pandas.mean(), inplace=True)
    X_train_norm = X_train_pandas.values
    X_test_norm = X_test_pandas.values
    X_val_norm = X_val_pandas.values

    X_train_norm = X_train_norm.astype(np.float32)
    X_test_norm = X_test_norm.astype(np.float32)
    X_val_norm = X_val_norm.astype(np.float32)
    y_train_ = y_train_.astype(np.int32)
    test_y = y_test.astype(np.int32)
    val_y = y_val.astype(np.int32)
    data_x_ = X_train_norm
    data_y_ = y_train_
    test_x = X_test_norm
    val_x = X_val_norm

    x_normal_train = data_x_[np.nonzero(y_train_ == 0)[0]]
    x_normal_val = val_x[np.nonzero(val_y == 0)[0]]
    traindata_normal_generator = DatasetGenerator(
        x_normal_train, x_normal_train)
    valdata_normal_generator = DatasetGenerator(x_normal_val, x_normal_val)
    traindata_generator = DatasetGenerator(data_x_, data_x_)
    testdata_generator = DatasetGenerator(test_x, test_x)
    valdata_generator = DatasetGenerator(val_x, val_x)

    dataset_train = ds.GeneratorDataset(traindata_generator, ["data", "label"],\
    shuffle=False).batch(1024, drop_remainder=True)
    dataset_test = ds.GeneratorDataset(testdata_generator, ["data", "label"],\
    shuffle=False).batch(1024, drop_remainder=True)
    dataset_val = ds.GeneratorDataset(valdata_generator, ["data", "label"],\
    shuffle=False).batch(1024, drop_remainder=True)
    dataset_normal_train = ds.GeneratorDataset(traindata_normal_generator,\
    ["data", "label"], shuffle=False).batch(1024, drop_remainder=True)
    dataset_normal_val = ds.GeneratorDataset(valdata_normal_generator,\
    ["data", "label"], shuffle=False).batch(1024, drop_remainder=True)

    input_dim_ = data_x_.shape[1]
    train_autoencoder(args_, dataset_normal_train, dataset_normal_val, input_dim_)
    draw_autoencoder(data_x_, data_y_, 0.2, f"./autoEncoder/{args_.data_name}_best_mse.ckpt", "scores.png")
    y_train_real = data_y_.copy()
    print('Abnormal samples in train_set before autoEncoder selection:', y_train_real.sum())
    data_y_ = select_autoencoder(
        data_x_, data_y_, args_.noiseLabelRate, f"./autoEncoder/{args_.data_name}_best_mse.ckpt")
    print('Abnormal samples in train_set after autoEncoder selection:', data_y_.sum())
