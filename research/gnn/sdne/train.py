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
"""
python train.py
"""
import os
import argparse
import random
import numpy as np

import mindspore
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback

from src import GraphDataset
from src import SDNE, SDNEWithLossCell, SDNELoss1, SDNELoss2
from src import initializer, optimizer
from src import check_reconstruction
from src import reconstruction_precision_k
from src import cfg

if cfg.is_modelarts:
    import moxing as mox
    DATA_URL = "/cache/data/"
    CKPT_URL = "/cache/ckpt/"
    TMP_URL = "/cache/tmp/"

parser = argparse.ArgumentParser(description='Mindspore SDNE Training')

# Datasets
parser.add_argument('--train_url', type=str, default='./train', help="train path")
parser.add_argument('--ckpt_url', type=str, default='./ckpt', help="ckpt path")
parser.add_argument('--data_url', type=str, default='', help='dataset path')
parser.add_argument('--data_path', type=str, default='', help='data path')
parser.add_argument('--label_path', type=str, default='', help='label path')
parser.add_argument('--dataset', type=str, default='WIKI',
                    choices=['WIKI', 'GRQC'])

# Optimization options
parser.add_argument('--epochs', type=int, default=40)

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--pretrained', type=bool, default=False, help='use pre-trained model')

# Device options
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend")
parser.add_argument('--device_id', type=int, default=0)


class EvalCallBack(Callback):
    """
    Precision verification using callback function.
    """
    # define the operator required and config
    def __init__(self, ds, c, args, lp='', tmp_dir='./tmp/'):
        super(EvalCallBack, self).__init__()
        self.ds = ds
        self.gemb = c['generate_emb']
        self.rec = c['reconstruction']
        self.clf = c['classify']
        self.batch = c['batch']
        self.label_path = lp
        self.tmp_dir = tmp_dir
        self.args = args

    # define operator function after finishing train
    def end(self, run_context):
        """
        eval function
        """
        cb_param = run_context.original_args()
        backbone = cb_param.train_network.network
        graph = self.ds.get_graph()
        _, data = self.ds.get_data()
        idx2node = self.ds.get_idx2node()
        embeddings = None
        if self.rec['check']:
            if self.args.dataset == 'WIKI':
                reconstructions, vertices = backbone.get_reconstructions(data, idx2node)
                reconstruction_precision_k(reconstructions, vertices, graph, self.rec['k_query'])
            else:
                embeddings = backbone.get_embeddings(data, self.batch)
                check_reconstruction(embeddings, graph, idx2node, self.rec['k_query'])
        if self.gemb:
            print('Storing embeddings data...')
            if embeddings is None:
                embeddings = backbone.get_embeddings(data, self.batch)
            np.save(self.tmp_dir + "embeddings.npy", embeddings)
            np.save(self.tmp_dir + "embeddings_idx.npy", idx2node)

def count_params(n):
    """
    count param
    """
    total_param = 0
    for param in n.trainable_params():
        total_param += np.prod(param.shape)
    return total_param

def run_train():
    args = parser.parse_args()
    config = cfg[args.dataset]

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    # set all paths
    ckpt_url = ''
    tmp_url = ''
    data_path = ''
    label_path = ''
    if cfg.is_modelarts:
        if not os.path.exists(DATA_URL):
            os.makedirs(DATA_URL, 0o755)
        mox.file.copy_parallel(args.data_url, DATA_URL)
        print("data finish copy to %s." % DATA_URL)
        data_path = os.path.join(DATA_URL, (config['data_path'] if args.data_path == '' else args.data_path))
        label_path = os.path.join(DATA_URL, (config['label_path'] if args.label_path == '' else args.label_path))
        if not os.path.exists(CKPT_URL):
            os.makedirs(CKPT_URL, 0o755)
        ckpt_url = CKPT_URL
        if not os.path.exists(TMP_URL):
            os.makedirs(TMP_URL, 0o755)
        tmp_url = TMP_URL
    else:
        if args.data_url == '':
            data_path = args.data_path
            label_path = args.label_path
        else:
            data_path = os.path.join(args.data_url, (config['data_path'] if args.data_path == '' else args.data_path))
            label_path = os.path.join(args.data_url,
                                      (config['label_path'] if args.label_path == '' else args.label_path))
        tmp_url = args.train_url
        ckpt_url = args.ckpt_url

    # read dataset
    dataset = GraphDataset(args.dataset, data_path, batch=config['batch'], delimiter=config['delimiter'])

    # initialize the SDNE model
    loss = None
    if args.dataset == 'WIKI':
        loss = SDNELoss1(alpha=config['alpha'], beta=config['beta'], gamma=config['gamma'])
    else:
        loss = SDNELoss2(alpha=config['alpha'], beta=config['beta'], gamma=config['gamma'])
    net = SDNEWithLossCell(SDNE(dataset.get_node_size(), hidden_size=config['hidden_size'], act=config['act'],
                                weight_init=initializer(config['weight_init'])), loss)
    if args.pretrained:
        param_dict = load_checkpoint(args.checkpoint)
        load_param_into_net(net, param_dict)
    optim = optimizer(net, config['optim'], args.epochs, dataset.get_data_size())
    model = Model(net, optimizer=optim)
    print('param num: ', count_params(net))

    config_ck = CheckpointConfig(save_checkpoint_steps=config['ckpt_step'], keep_checkpoint_max=config['ckpt_max'])
    ckpoint_cb = ModelCheckpoint(prefix="SDNE_" + args.dataset, config=config_ck, directory=ckpt_url)
    time_cb = TimeMonitor(data_size=dataset.get_node_size())
    loss_cb = LossMonitor()
    eval_cb = EvalCallBack(dataset, config, args, label_path, tmp_url)
    cb = [ckpoint_cb, time_cb, loss_cb, eval_cb]

    model.train(args.epochs, dataset.get_dataset(), callbacks=cb, dataset_sink_mode=False)

    if cfg.is_modelarts:
        mox.file.copy_parallel(CKPT_URL, args.train_url)
        mox.file.copy_parallel(TMP_URL, args.train_url)


if __name__ == "__main__":
    # fix all random seed
    mindspore.set_seed(1)
    np.random.seed(1)
    random.seed(1)

    run_train()
