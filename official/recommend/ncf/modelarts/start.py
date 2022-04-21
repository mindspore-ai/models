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
"""Training entry file"""
import os

import argparse
import ast
import glob
import moxing as mox
import numpy as np
from absl import logging

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, export, Model, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.common import set_seed

from dataset import create_dataset
from ncf import NCFModel, NetWithLossClass, TrainStepWrap, PredictWithSigmoid
import src.constants as rconst
from src.config import cfg
from src.metrics import NCFMetric

set_seed(1)

logging.set_verbosity(logging.INFO)

# modelarts modification------------------------------
CACHE_TRAINING_URL = "/cache/training/"
CKPT_SAVE_DIR = CACHE_TRAINING_URL
REAL_PATH = '/cache/dataset'

if not os.path.isdir(CACHE_TRAINING_URL):
    os.makedirs(CACHE_TRAINING_URL)
# modelarts modification------------------------------

parser = argparse.ArgumentParser(description='NCF')
parser.add_argument("--data_path", type=str, default="./dataset/")  # The location of the input data.
parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "ml-20m"])  # Dataset to be trained and evaluated. ["ml-1m", "ml-20m"]
parser.add_argument("--train_epochs", type=int, default=14)  # The number of epochs used to train.
parser.add_argument("--batch_size", type=int, default=256)  # Batch size for training and evaluation
parser.add_argument("--num_neg", type=int, default=4)  # The Number of negative instances to pair with a positive instance.
parser.add_argument("--output_path", type=str, default="./output/")  # The location of the output file.
parser.add_argument("--loss_file_name", type=str, default="loss.log")  # Loss output file.
parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/")  # The location of the checkpoint file.
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: None)')
parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

parser.add_argument("--data_url", type=str, default="/NCF/dataset", help="path to dataset")
parser.add_argument("--train_url", type=str, default="/NCF/output", help="path of training output")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument("--filter_weight", type=ast.literal_eval, default=True,
                    help="Filter head weight parameters, default is False.")
parser.add_argument("--is_row_vector_input", type=ast.literal_eval, default=True,
                    help="Change model input into row vector for MindX SDK inference")
args = parser.parse_args()

def test_train():
    """train entry method"""
    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
            context.set_context(device_id=args.device_id)
        elif args.device_target == "GPU":
            init()

        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, gradients_mean=True)
    else:
        context.set_context(device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    layers = cfg.layers
    num_factors = cfg.num_factors
    epochs = args.train_epochs

    # modelarts modification------------------------------
    if not os.path.exists(REAL_PATH):
        os.makedirs(REAL_PATH, 0o755)
    mox.file.copy_parallel(args.data_url, REAL_PATH)
    print("training data finish copy to %s." % REAL_PATH)

    ds_train, num_train_users, num_train_items = create_dataset(test_train=True, data_dir=REAL_PATH,
                                                                dataset=args.dataset, train_epochs=1,
                                                                batch_size=args.batch_size, num_neg=args.num_neg,
                                                                row_vector=args.is_row_vector_input)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    # modelarts modification------------------------------

    ncf_net = NCFModel(num_users=num_train_users,
                       num_items=num_train_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16,
                       is_row_vector_input=args.is_row_vector_input)

    # modelarts modification------------------------------
    if args.pre_trained:
        model_file_name = args.pre_trained.split(os.path.sep)[-1]
        print("---------------------------------mdoel file name")
        print(model_file_name)
        cache_model_file_path = os.path.join(REAL_PATH, model_file_name)
        param_dict = load_checkpoint(cache_model_file_path)

        if args.filter_weight:
            filter_list = [x.name for x in ncf_net.logits_dense.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)

        load_param_into_net(ncf_net, param_dict)
    # modelarts modification------------------------------

    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net, ds_train.get_dataset_size() * (epochs + 1))

    train_net.set_train()

    model = Model(train_net)
    callback = LossMonitor(per_print_times=ds_train.get_dataset_size())
    ckpt_config = CheckpointConfig(save_checkpoint_steps=(4970845+args.batch_size-1)//(args.batch_size),
                                   keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix='ncf', directory=CKPT_SAVE_DIR, config=ckpt_config)
    model.train(epochs,
                ds_train,
                callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb],
                dataset_sink_mode=True)

    # modelarts modification------------------------------
    ckpt_list = glob.glob("/cache/training/ncf*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    ncf_net = NCFModel(num_users=num_train_users,
                       num_items=num_train_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16,
                       is_row_vector_input=args.is_row_vector_input)

    frozen_to_air_args = {'ckpt_file': ckpt_model,
                          'topk': rconst.TOP_K,
                          'num_eval_neg': rconst.NUM_EVAL_NEGATIVES,
                          'file_name': '/cache/training/ncf',
                          'file_format': 'AIR'}
    frozen_to_air(ncf_net, frozen_to_air_args)
    # modelarts modification------------------------------

def frozen_to_air(net, args_net):
    """frozen net parameters in the format of air"""
    param_dict = load_checkpoint(args_net.get("ckpt_file"))
    load_param_into_net(net, param_dict)

    network = PredictWithSigmoid(net, args_net.get("topk"), args_net.get("num_eval_neg"))

    users = Tensor(np.zeros([1, cfg.eval_batch_size]).astype(np.int32))
    items = Tensor(np.zeros([1, cfg.eval_batch_size]).astype(np.int32))
    masks = Tensor(np.zeros([1, cfg.eval_batch_size]).astype(np.float32))
    input_data = [users, items, masks]
    export(network, *input_data, file_name=args_net.get("file_name"), file_format=args_net.get("file_format"))

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def test_eval():
    """eval method"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Davinci",
                        save_graphs=True,
                        device_id=args.device_id)

    layers = cfg.layers
    num_factors = cfg.num_factors
    topk = rconst.TOP_K
    num_eval_neg = rconst.NUM_EVAL_NEGATIVES

    ds_eval, num_eval_users, num_eval_items = create_dataset(test_train=False, data_dir=REAL_PATH,
                                                             dataset=args.dataset, train_epochs=0,
                                                             eval_batch_size=cfg.eval_batch_size,
                                                             row_vector=args.is_row_vector_input)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    ncf_net = NCFModel(num_users=num_eval_users,
                       num_items=num_eval_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16,
                       is_row_vector_input=args.is_row_vector_input)

    ckpt_list = glob.glob("/cache/training/ncf*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    checkpoint_path = os.path.join(CKPT_SAVE_DIR, ckpt_model)
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(ncf_net, param_dict)

    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(ncf_net, topk, num_eval_neg)

    ncf_metric = NCFMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"ncf": ncf_metric})

    ncf_metric.clear()
    out = model.eval(ds_eval)

    eval_file_path = os.path.join(CACHE_TRAINING_URL, "result.txt")
    eval_file = open(eval_file_path, "a+")
    eval_file.write("EvalCallBack: HR = {}, NDCG = {}\n".format(out['ncf'][0], out['ncf'][1]))
    eval_file.close()
    print("EvalCallBack: HR = {}, NDCG = {}".format(out['ncf'][0], out['ncf'][1]))

    mox.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)

if __name__ == '__main__':
    test_train()
    test_eval()
