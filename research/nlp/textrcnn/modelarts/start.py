# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#################train textcnn example on movie review########################
python train.py
"""
import os
import sys
import shutil
import numpy as np

import moxing as mox
import mindspore.nn as nn
import mindspore.context as context
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore import Tensor, load_checkpoint, load_param_into_net, export

from src.dataset import create_dataset
from src.dataset import convert_to_mindrecord
from src.textrcnn import textrcnn
from src.utils import get_lr
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config as cfg
from src.model_utils.device_adapter import get_device_id

set_seed(2)
os.system("pip3 install gensim==4.0.1 python-Levenshtein urllib3==1.26.5 chardet==3.0.4")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def convert_encoding(path_name, file_name):
    """convert encoding method"""
    f = open(file_name, 'r', encoding='iso-8859-1')
    tmp_name = os.path.join(path_name, "tmp")
    f_tmp = open(tmp_name, 'w+', encoding='utf-8')
    for line in f:
        f_tmp.write(line)
    for line in f_tmp:
        print(line)
    f.close()
    f_tmp.close()
    os.remove(file_name)
    os.rename(tmp_name, file_name)


def dataset_split(label):
    """dataset_split api"""
    # label can be 'pos' or 'neg'
    pos_samples = []
    pos_path = os.path.join(cfg.data_url, "rt-polaritydata")
    pos_file = os.path.join(pos_path, "rt-polarity." + label)

    convert_encoding(pos_path, pos_file)

    pfhand = open(pos_file, encoding='utf-8')
    pos_samples += pfhand.readlines()
    pfhand.close()
    np.random.seed(0)
    perm = np.random.permutation(len(pos_samples))
    perm_train = perm[0:int(len(pos_samples) * 0.9)]
    perm_test = perm[int(len(pos_samples) * 0.9):]
    pos_samples_train = []
    pos_samples_test = []
    for pt in perm_train:
        print(pos_samples[pt])
        pos_samples_train.append(pos_samples[pt])
    for pt in perm_test:
        pos_samples_test.append(pos_samples[pt])

    if not os.path.exists(os.path.join(cfg.data_url, 'train')):
        os.makedirs(os.path.join(cfg.data_url, 'train'))
    if not os.path.exists(os.path.join(cfg.data_url, 'test')):
        os.makedirs(os.path.join(cfg.data_url, 'test'))

    f = open(os.path.join(cfg.data_url, 'train', label), "w")
    f.write(''.join(pos_samples_train))
    f.close()

    f = open(os.path.join(cfg.data_url, 'test', label), "w")
    f.write(''.join(pos_samples_test))
    f.close()


def modelarts_pre_process():
    """modelarts pre process function."""
    # checkpoint files were saved at "os.path.join(cfg.output_path, cfg.ckpt_folder_path)"
    cfg.ckpt_folder_path = os.path.join(cfg.output_path, cfg.ckpt_folder_path)
    cfg.file_name = os.path.join(cfg.ckpt_folder_path, cfg.file_name)
    cfg.ckpt_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), cfg.ckpt_file)
    cfg.preprocess_path = os.path.join(cfg.data_url, cfg.preprocess_path)
    cfg.pre_result_path = os.path.join(cfg.data_url, cfg.pre_result_path)
    cfg.emb_path = os.path.join(cfg.data_url, cfg.emb_path)
    cfg.data_root = cfg.data_url
    print(cfg.data_url)


def run_train():
    """train function."""
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="Ascend",
        device_id=get_device_id())

    if cfg.preprocess == 'true':
        print("============== Starting Data Pre-processing ==============")
        if os.path.exists(cfg.preprocess_path):
            shutil.rmtree(cfg.preprocess_path)
        os.mkdir(cfg.preprocess_path)
        convert_to_mindrecord(cfg.embed_size, cfg.data_root, cfg.preprocess_path, cfg.emb_path)
        print("============== Saved to " + cfg.preprocess_path + "==============")

    if cfg.cell == "vanilla":
        print("============== Precision is lower than expected when using vanilla RNN architecture ==============")

    embedding_table = np.loadtxt(os.path.join(cfg.preprocess_path, "weight.txt")).astype(np.float32)

    network = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                       cell=cfg.cell, batch_size=cfg.batch_size)

    ds_train = create_dataset(cfg.preprocess_path, cfg.batch_size, True)
    step_size = ds_train.get_dataset_size()

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    lr = get_lr(cfg, step_size)
    num_epochs = cfg.num_epochs
    if cfg.cell == "lstm":
        num_epochs = cfg.lstm_num_epochs

    opt = nn.Adam(params=network.trainable_params(), learning_rate=lr)

    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    model = Model(network, loss, opt, {'acc': Accuracy()}, amp_level="O3")

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.cell, directory=cfg.ckpt_folder_path, config=config_ck)
    model.train(num_epochs, ds_train, callbacks=[ckpoint_cb, loss_cb, time_cb])
    print("train success")


def run_eval():
    """eval function."""
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="Ascend",
        device_id=get_device_id())

    embedding_table = np.loadtxt(os.path.join(cfg.preprocess_path, "weight.txt")).astype(np.float32)
    print("============== Read from " + cfg.preprocess_path + "==============")
    network = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                       cell=cfg.cell, batch_size=cfg.batch_size)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    eval_net = nn.WithEvalCell(network, loss, True)
    # loss_cb = LossMonitor()
    print("============== Starting Testing ==============")
    ds_eval = create_dataset(cfg.preprocess_path, cfg.batch_size, False)

    num_epochs = cfg.num_epochs
    if cfg.cell == "lstm":
        num_epochs = cfg.lstm_num_epochs
    ckpt_file = cfg.cell + "-" + str(num_epochs) + "_" + str(cfg.save_checkpoint_steps) + ".ckpt"
    cfg.ckpt_path = os.path.join(cfg.ckpt_folder_path, ckpt_file)
    param_dict = load_checkpoint(cfg.ckpt_path)
    load_param_into_net(network, param_dict)
    network.set_train(False)
    model = Model(network, loss, metrics={'acc': Accuracy()}, eval_network=eval_net, eval_indexes=[0, 1, 2])
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


def run_export():
    """export function."""
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="Ascend",
        device_id=get_device_id())

    # define net
    embedding_table = np.loadtxt(os.path.join(cfg.preprocess_path, "weight.txt")).astype(np.float32)
    print("============== Read from " + cfg.preprocess_path + "==============")
    net = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                   cell=cfg.cell, batch_size=cfg.batch_size)

    # load checkpoint
    num_epochs = cfg.num_epochs
    if cfg.cell == "lstm":
        num_epochs = cfg.lstm_num_epochs
    ckpt_file = cfg.cell + "-" + str(num_epochs) + "_" + str(cfg.save_checkpoint_steps) + ".ckpt"
    cfg.ckpt_file = os.path.join(cfg.ckpt_folder_path, ckpt_file)
    param_dict = load_checkpoint(cfg.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    image = Tensor(np.ones([cfg.batch_size, 50], np.int32))
    export(net, image, file_name=cfg.file_name, file_format=cfg.file_format)


def get_bin():
    """generate bin files."""
    ds_eval = create_dataset(cfg.preprocess_path, cfg.batch_size, False)
    img_path = os.path.join(cfg.pre_result_path, "00_feature")
    os.makedirs(img_path)
    label_list = []

    for i, data in enumerate(ds_eval.create_dict_iterator(output_numpy=True)):
        file_name = "textrcnn_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(img_path, file_name)

        data["feature"].tofile(file_path)
        label_list.append(data["label"])

    np.save(os.path.join(cfg.pre_result_path, "label_ids.npy"), label_list)
    print("=" * 20, "bin files finished", "=" * 20)


@moxing_wrapper(pre_process=modelarts_pre_process)
def start():
    """main process function"""
    dataset_split('pos')
    dataset_split('neg')

    run_train()
    run_eval()
    run_export()

    get_bin()
    mox.file.copy_parallel(cfg.pre_result_path, "obs://" + cfg.bucket_name + "/data/pre_result_path")


if __name__ == '__main__':
    start()
