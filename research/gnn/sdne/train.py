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
python train.py
"""
import os
import argparse
import random
import numpy as np

import mindspore
from mindspore import nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback

from src import GraphDataset
from src import SDNE, SDNEWithLossCell, SDNELoss
from src import reconstruction_precision_k
from src import cfg

if cfg.is_modelarts:
    import moxing as mox

DATA_PATH = "/cache/data/"
CKPT_PATH = "/cache/ckpt/"
TMP_PATH = "/cache/tmp/"

parser = argparse.ArgumentParser(description='Mindspore SDNE Training')

# Datasets
parser.add_argument('--train_url', type=str, default='./train', help="train path")
parser.add_argument('--ckpt_url', type=str, default='./ckpt', help="ckpt path")
parser.add_argument('--data_url', type=str, default='', help='data path')
parser.add_argument('--dataset', type=str, default='WIKI',
                    choices=['WIKI', 'BLOGCATALOG', 'FLICKR', 'YOUTUBE', 'GRQC', 'NEWSGROUP'])

# Optimization options
parser.add_argument('--epochs', type=int, default=40)

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--pretrained', type=bool, default=False, help='use pre-trained model')

# Device options
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()

class EvalCallBack(Callback):
    """
    Precision verification using callback function.
    """
    # define the operator required and config
    def __init__(self, ds, eval_cfg, gemb=False, grec=False, tmp_dir='./tmp/'):
        super(EvalCallBack, self).__init__()
        self.ds = ds
        self.frac = eval_cfg['frac']
        self.use_rand = eval_cfg['use_rand']
        self.k_query = eval_cfg['k_query']
        self.gemb = gemb
        self.grec = grec
        self.tmp_dir = tmp_dir

    # define operator function after network training
    def end(self, run_context):
        """
        eval function
        """
        cb_param = run_context.original_args()
        backbone = cb_param.train_network.network.network
        graph = self.ds.get_graph()
        index, data = self.ds.get_data(self.frac, self.use_rand)
        idx2node_y = self.ds.get_idx2node()
        idx2node_x = idx2node_y[index]
        reconstructions, vertices = backbone.get_reconstructions(data, idx2node_y, idx2node_x)
        reconstruction_precision_k(reconstructions, vertices, graph, self.k_query)
        if self.grec:
            print('Storing reconstruction data')
            np.save(self.tmp_dir + "reconstruction.npy", reconstructions)
            np.save(self.tmp_dir + "reconstruction_idx.npy", vertices)
        if self.gemb:
            print('Storing embeddings data')
            embeddings = backbone.get_embeddings(data)
            np.save(self.tmp_dir + "embeddings.npy", embeddings)
            np.save(self.tmp_dir + "embeddings_idx.npy", idx2node_x)

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)

    config = cfg[args.dataset]

    # fix all random seed
    mindspore.set_seed(1)
    np.random.seed(1)
    random.seed(1)

    data_url = args.data_url
    ckpt_url = args.ckpt_url
    tmp_url = "./"
    if cfg.is_modelarts:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH, 0o755)
        mox.file.copy_parallel(args.data_url, DATA_PATH)
        print("data finish copy to %s." % DATA_PATH)
        data_url = DATA_PATH + config['dataset_path']
        if not os.path.exists(CKPT_PATH):
            os.makedirs(CKPT_PATH, 0o755)
        ckpt_url = CKPT_PATH
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH, 0o755)
        tmp_url = TMP_PATH

    if data_url == '':
        data_url = config['dataset_path']
    dataset = GraphDataset(data_url, batch=config['batch'],
                           delimiter=config['delimiter'], linkpred=config['linkpred'])
    net = SDNEWithLossCell(SDNE(dataset.get_node_size(), hidden_size=config['hidden_size'],
                                weight_init=config['weight_init']),
                           SDNELoss(alpha=config['alpha'], beta=config['beta']))
    if args.pretrained:
        param_dict = load_checkpoint(args.checkpoint)
        load_param_into_net(net, param_dict)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=config['learning_rate'],
                    weight_decay=config['weight_decay'])
    model = Model(net, optimizer=optim)

    config_ck = CheckpointConfig(save_checkpoint_steps=config['ckpt_step'], keep_checkpoint_max=config['ckpt_max'])
    ckpoint_cb = ModelCheckpoint(prefix="SDNE", config=config_ck, directory=ckpt_url)
    time_cb = TimeMonitor(data_size=dataset.get_node_size())
    loss_cb = LossMonitor()
    eval_cb = EvalCallBack(dataset, config['eval'], config['generate_emb'], config['generate_rec'], tmp_dir=tmp_url)
    cb = [ckpoint_cb, time_cb, loss_cb, eval_cb]

    model.train(args.epochs, dataset.get_dataset(), callbacks=cb)

    if cfg.is_modelarts:
        mox.file.copy_parallel(CKPT_PATH, args.train_url)
        mox.file.copy_parallel(TMP_PATH, args.train_url)
