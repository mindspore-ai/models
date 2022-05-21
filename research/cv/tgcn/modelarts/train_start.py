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
Training script
"""
import os
import argparse
import time
import numpy as np
from mindspore.communication import init, get_rank
from mindspore.context import ParallelMode
from mindspore import dtype as mstype
from mindspore import export, set_seed, nn, context, Model, load_checkpoint, load_param_into_net, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor, Callback
from mindspore import save_checkpoint
from mindspore.dataset.core.validator_helpers import INT32_MAX
from src.config import ConfigTGCN
from src.dataprocess import load_adj_matrix, load_feat_matrix, generate_dataset_ms, generate_dataset_ms_distributed
from src.task import SupervisedForecastTask
from src.model.loss import TGCNLoss
from src.callback import RMSE


class SaveCallback(Callback):
    """
    Save the best checkpoint (minimum RMSE) during training
    """

    def __init__(self, eval_model, ds_eval, config):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.rmse = INT32_MAX
        self.config = config

    def epoch_end(self, run_context):
        """Evaluate the network and save the best checkpoint (minimum RMSE)"""
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        cb_params = run_context.original_args()
        file_name = self.config.dataset + '_' + str(self.config.pre_len) + '.ckpt'
        if self.config.save_best:
            result = self.model.eval(self.ds_eval)
            print('Eval RMSE:', '{:.6f}'.format(result['RMSE']))
            if result['RMSE'] < self.rmse:
                self.rmse = result['RMSE']
                save_checkpoint(save_obj=cb_params.train_network,
                                ckpt_file_name=os.path.join(self.config.train_url, file_name))
                print("Best checkpoint saved!")
        else:
            save_checkpoint(save_obj=cb_params.train_network,
                            ckpt_file_name=os.path.join(self.config.train_url, file_name))


def _export(config):
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device)
    # Create network
    adj = load_adj_matrix(config.dataset, config.data_url)
    net = SupervisedForecastTask(adj, config.hidden_dim, config.pre_len)
    # Load parameters from checkpoint into network
    ckpt_file = config.dataset + "_" + str(config.pre_len) + ".ckpt"
    param_dict = load_checkpoint(os.path.join(config.train_url, ckpt_file))
    print(os.path.join(config.train_url, ckpt_file))
    print(param_dict)
    load_param_into_net(net, param_dict)
    # Initialize dummy inputs
    inputs = np.random.uniform(0.0, 1.0, size=[config.batch_size, config.seq_len, adj.shape[0]]).astype(np.float32)
    # Export network into MINDIR model file
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    file_name = config.dataset + "_" + str(config.pre_len)
    path = os.path.join(config.train_url, file_name)
    # export(net, Tensor(inputs), file_name=path, file_format='ONNX')
    export(net, Tensor(inputs), file_name=path, file_format='AIR')
    print("==========================================")
    # print(file_name + ".onnx exported successfully!")
    print(file_name + ".air exported successfully!")
    print("==========================================")

def merge_cfg(t_args, config):
    for arg in vars(t_args):
        setattr(config, arg, getattr(t_args, arg))
    return config

def run_train(args):
    """
    Run training
    """
    # Config initialization
    config = ConfigTGCN()
    # Set global seed for MindSpore and NumPy
    set_seed(config.seed)
    # ModelArts runtime, datasets and network initialization
    config = merge_cfg(args, config)
    print(config.data_url)
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=device_id)

    if config.distributed:
        device_num = int(os.getenv('RANK_SIZE'))
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        training_set = generate_dataset_ms_distributed(config, training=True, abs_path=config.data_url)
        eval_set = generate_dataset_ms_distributed(config, training=False, abs_path=config.data_url)
        _, max_val = load_feat_matrix(config.dataset, config.data_url)
        net = SupervisedForecastTask(load_adj_matrix(config.dataset, config.data_url),
                                     config.hidden_dim, config.pre_len)
    else:
        training_set = generate_dataset_ms(config, training=True, abs_path=config.data_url)
        eval_set = generate_dataset_ms(config, training=False, abs_path=config.data_url)
        _, max_val = load_feat_matrix(config.dataset, abs_path=config.data_url)
        net = SupervisedForecastTask(load_adj_matrix(config.dataset, abs_path=config.data_url),
                                     config.hidden_dim, config.pre_len)

    # Mixed precision
    net.tgcn.tgcn_cell.graph_conv1.matmul.to_float(mstype.float16)
    net.tgcn.tgcn_cell.graph_conv2.matmul.to_float(mstype.float16)
    # Loss function
    loss_fn = TGCNLoss()
    # Optimizer
    optimizer = nn.Adam(net.trainable_params(), config.learning_rate, weight_decay=config.weight_decay)
    # Create model
    model = Model(net, loss_fn, optimizer, {'RMSE': RMSE(max_val)})
    # Training
    time_start = time.time()
    callbacks = [LossMonitor(), TimeMonitor()]
    if config.distributed:
        print("==========Distributed Training Start==========")
        save_callback = SaveCallback(model, eval_set, config)
        if get_rank() == 0:
            callbacks = [LossMonitor(), TimeMonitor(), save_callback]
        elif config.save_best:
            callbacks = [LossMonitor()]
    else:
        print("==========Training Start==========")
        save_callback = SaveCallback(model, eval_set, config)
        callbacks.append(save_callback)
    model.train(config.epochs, training_set,
                callbacks=callbacks,
                dataset_sink_mode=config.data_sink)
    time_end = time.time()
    if config.distributed:
        print("==========Distributed Training End==========")
    else:
        print("==========Training End==========")
    print("Training time in total:", '{:.6f}'.format(time_end - time_start), "s")
    _export(config)

if __name__ == '__main__':
    # Set universal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
    parser.add_argument('--distributed', help="distributed training", type=bool, default=False)
    # Set ModelArts arguments
    parser.add_argument('--data_url', help='ModelArts location of data', type=str, default=None)
    parser.add_argument('--train_url', help='ModelArts location of training outputs', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SZ-taxi')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--pre_len', type=int, default=1)
    run_args = parser.parse_args()
    # Training
    run_train(run_args)
