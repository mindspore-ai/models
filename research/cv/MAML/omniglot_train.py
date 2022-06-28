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

import argparse
import os
import numpy as np
from src.OmniglotIter import IterDatasetGenerator
from src.meta import Meta
import mindspore.context as context
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import load_checkpoint

def main(args):
    np.random.seed(222)
    print(args)
    config = [
        ('conv2d', [1, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('reduce_mean', []),
        # ('flatten', []),
        ('linear', [64, args.n_way])
    ]
    if args.ckpt != '':
        param_dict = load_checkpoint(args.ckpt)
        maml = Meta(args, config, param_dict)
    else:
        maml = Meta(args, config)

    context.set_context(device_id=args.device_id)

    db_train = IterDatasetGenerator(args.data_path,
                                    batchsz=args.task_num,
                                    n_way=args.n_way,
                                    k_shot=args.k_spt,
                                    k_query=args.k_qry,
                                    imgsz=args.imgsz,
                                    itera=1)
    config = CheckpointConfig(save_checkpoint_steps=1,
                              keep_checkpoint_max=2000,
                              saved_network=maml)
    ckpoint_cb = ModelCheckpoint(prefix='maml', directory=args.output_dir, config=config)

    inp = ds.GeneratorDataset(db_train, ['x_spt', 'y_spt', 'x_qry', 'y_qry'])

    maml.set_grad(True)
    model = Model(maml)
    model.train(args.epoch, inp, callbacks=[TimeMonitor(), LossMonitor(), ckpoint_cb], dataset_sink_mode=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device_id', type=int, help='device id', default=1)
    argparser.add_argument('--device_target', type=str, help='device target', default='GPU')
    argparser.add_argument('--mode', type=str, help='pynative or graph', default='graph')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--lr_scheduler_gamma', type=float, help='update steps for finetunning', default=0.5)
    argparser.add_argument('--output_dir', type=str, help='update steps for finetunning', default='./ckpt_outputs')
    argparser.add_argument('--ckpt', type=str, help='trained model', default='')
    argparser.add_argument('--data_path', type=str, help='path of data', default='/your/path/omniglot/')
    arg = argparser.parse_args()
    if arg.mode == 'pynative':
        context.set_context(mode=context.PYNATIVE_MODE)
    elif arg.mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE)
    if os.path.exists('loss.txt'):
        os.remove('loss.txt')
    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir)
    main(arg)
